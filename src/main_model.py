import numpy as np
import torch
import torch.nn as nn
from src.diff_model import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = config["diffusion"].get("target_dim", target_dim)

        config_diff = config["diffusion"]
        self.diffmodel = diff_CSDI(config_diff, inputdim=self.target_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def calc_loss_valid(
        self,
        treatment,
        outcomes,
        outcomes_mask,
        x_seq,
        is_train,
        propnet,
        x_prop=None,
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                treatment,
                outcomes,
                outcomes_mask,
                x_seq,
                is_train,
                propnet=propnet,
                set_t=t,
                x_prop=x_prop,
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        treatment,
        outcomes,
        outcomes_mask,
        x_seq,
        is_train,
        propnet=None,
        set_t=-1,
        x_prop=None,
    ):
        B = outcomes.shape[0]
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(outcomes)
        noisy_data = (current_alpha**0.5) * outcomes + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        # diff_CSDI conditions on covariate sequence x_seq and global (treatment, noisy_target); see diff_model.diff_CSDI.forward
        predicted = self.diffmodel(
            x_seq=x_seq,
            treatment=treatment,
            noisy_target=noisy_data,
            diffusion_step=t,
        ).to(self.device)

        residual = (noise - predicted) * outcomes_mask
        num_eval = outcomes_mask.sum()

        x_batch = x_prop if x_prop is not None else x_seq.reshape(B, -1)
        t_batch = treatment.reshape(-1)
        pi_hat = propnet.forward(x_batch.float())
        weights = (t_batch / pi_hat[:, 1]) + ((1 - t_batch) / pi_hat[:, 0])
        weights = torch.clamp(weights.reshape(-1, 1), min=0.1, max=0.9)
        # weights = torch.clamp(weights.reshape(-1, 1), min=0.1)

        loss = (weights * (residual**2)).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, treatment, outcomes, x_seq, n_samples):
        B = outcomes.shape[0]
        imputed_samples = torch.zeros(B, n_samples, 2).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(outcomes)

            for t_step in range(self.num_steps - 1, -1, -1):
                t_batch = torch.full((B,), t_step, device=self.device, dtype=torch.long)
                predicted = self.diffmodel(
                    x_seq=x_seq,
                    treatment=treatment,
                    noisy_target=current_sample,
                    diffusion_step=t_batch,
                ).to(self.device)

                coeff1 = 1 / self.alpha_hat[t_step] ** 0.5
                coeff2 = (1 - self.alpha_hat[t_step]) / (1 - self.alpha[t_step]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t_step > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t_step - 1])
                        / (1.0 - self.alpha[t_step])
                        * self.beta[t_step]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1, propnet = None):
        (
            treatment,
            outcomes,
            mu,
            x_seq,
            x_mask,
            outcomes_mask,
            observed_tp,
            x_prop,
        ) = self.process_data(batch)

        return self.calc_loss(
            treatment=treatment,
            outcomes=outcomes,
            outcomes_mask=outcomes_mask,
            x_seq=x_seq,
            is_train=is_train,
            propnet=propnet,
            set_t=-1,
            x_prop=x_prop,
        )

    def evaluate(self, batch, n_samples):
        (
            treatment,
            outcomes,
            mu,
            x_seq,
            x_mask,
            outcomes_mask,
            observed_tp,
            x_prop,
        ) = self.process_data(batch)

        with torch.no_grad():
            samples = self.impute(treatment, outcomes, x_seq, n_samples)

        return samples, outcomes, mu, treatment


class CoDiS(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(CoDiS, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        treatment = batch["treatment"].to(self.device).float()
        outcomes = batch["outcomes"].to(self.device).float()
        mu = batch["mu"].to(self.device).float()
        x_seq = batch["x_seq"].to(self.device).float()
        x_mask = batch["x_mask"].to(self.device).float()
        outcomes_mask = batch["outcomes_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        x_prop = (
            batch["x_prop"].to(self.device).float()
            if "x_prop" in batch
            else None
        )

        return (
            treatment,
            outcomes,
            mu,
            x_seq,
            x_mask,
            outcomes_mask,
            observed_tp,
            x_prop,
        )