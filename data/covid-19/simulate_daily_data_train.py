import torch
import numpy as np
import pandas as pd
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import math
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RNG_SEED = 42

# Outcome observation horizon (days) for causal samples. Use a daily 0..H grid (not just
# endpoints) so rk4 integration stays stable and beta differences propagate to deaths.
CAUSAL_FUTURE_DAYS = 30

# Bernoulli probability for background-pool mask_policy (decoupled from sample-level propensity).
CAUSAL_TREATMENT_ONE_PROB = 0.4

# City-level train/test split to avoid sliding-window leakage across splits.
CAUSAL_SPLIT_TRAIN_FRAC = 0.80
CAUSAL_SPLIT_SEED = RNG_SEED

# STATE_NAMES column indices for 14-day window summaries.
_IDX_IA, _IDX_IP, _IDX_IM, _IDX_IS = 2, 3, 4, 5
_IDX_HR, _IDX_HD, _IDX_D = 7, 8, 9


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def compute_window_stats(x_hist, pop: float):
    """
    Normalized epidemiological summaries from a 14-day history (per-capita).

    x_hist: shape [14, 10], columns in STATE_NAMES order; raw counts or pop-scaled.
    """
    x = np.asarray(x_hist, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(14, 10)
    pop_f = float(pop)
    infectious = x[:, _IDX_IA] + x[:, _IDX_IP] + x[:, _IDX_IM] + x[:, _IDX_IS]
    pressure = infectious[-1] / pop_f
    severe_burden = (x[-1, _IDX_IS] + x[-1, _IDX_HR] + x[-1, _IDX_HD]) / pop_f
    growth = (infectious[-1] - infectious[0]) / pop_f
    death_trend = (x[-1, _IDX_D] - x[0, _IDX_D]) / pop_f
    return pressure, severe_burden, growth, death_trend


def beta_for_arms_from_window(x_hist, y_anchor, pop: float):
    """
    Window-dependent beta for strict/relaxed arms (not anchor-only); increases ITE heterogeneity.
    y_anchor is kept for API compatibility; beta depends only on window summaries.
    """
    pressure, severe_burden, growth, death_trend = compute_window_stats(x_hist, pop)

    p = np.clip(pressure / 0.015, 0.0, 1.5)
    g = np.clip(growth / 0.006, -1.0, 1.5)
    h = np.clip(severe_burden / 0.004, 0.0, 1.5)

    beta_strict = 0.12 * (1.0 - 0.30 * p - 0.15 * h)
    beta_relaxed = 0.42 * (1.0 + 0.90 * p + 0.60 * g + 0.40 * h)

    beta_strict = float(np.clip(beta_strict, 0.05, 0.22))
    beta_relaxed = float(np.clip(beta_relaxed, 0.35, 1.20))
    return beta_strict, beta_relaxed


def treatment_propensity_from_window(x_hist, pop: float) -> float:
    """Confounded propensity from the 14-day window; clipped to [0.15, 0.85] for positivity."""
    pressure, severe_burden, growth, death_trend = compute_window_stats(x_hist, pop)

    z = (
        -0.25
        + 3.0 * np.clip(pressure / 0.015, 0.0, 2.0)
        + 2.0 * np.clip(severe_burden / 0.004, 0.0, 2.0)
        + 1.5 * np.clip(growth / 0.006, -1.0, 2.0)
    )

    e = sigmoid(z)
    return float(np.clip(e, 0.15, 0.85))


def add_outcome_noise(mu: float, x_hist, pop: float, arm: int) -> float:
    """Heteroscedastic noise on clean ODE outcome mu; relaxed arm (0) is noisier."""
    pressure, severe_burden, growth, death_trend = compute_window_stats(x_hist, pop)

    sigma = (
        0.03
        + 0.12 * np.clip(pressure / 0.015, 0.0, 2.0)
        + 0.08 * np.clip(severe_burden / 0.004, 0.0, 2.0)
    )

    if arm == 0:
        sigma *= 1.25

    y = mu + np.random.normal(0.0, sigma)
    return float(max(0.0, y))


def assign_city_splits(
    city_ids,
    train_frac: float = CAUSAL_SPLIT_TRAIN_FRAC,
    seed: int = CAUSAL_SPLIT_SEED,
) -> dict:
    """
    Split by city (y0_index) into train/test to avoid sample-level sliding-window leakage.
    Returns {city_id: 'train'|'test'}.
    """
    unique_cities = sorted(set(city_ids))
    rng = random.Random(seed)
    shuffled = unique_cities.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * train_frac)))
    if n_train >= len(shuffled) and len(shuffled) > 1:
        n_train = len(shuffled) - 1
    train_set = set(shuffled[:n_train])
    return {cid: ("train" if cid in train_set else "test") for cid in unique_cities}


def _print_causal_diagnostics(df: pd.DataFrame) -> None:
    """Print overlap, ITE, y/mu separation, and city-split diagnostics after generation."""
    ite = df["ite"].to_numpy(dtype=np.float64)
    abs_ite = np.abs(ite)

    print("\n=== Causal dataset diagnostics ===")
    if "propensity" in df.columns:
        e = df["propensity"].to_numpy(dtype=np.float64)
        print(
            f"Propensity: min={e.min():.4f}, max={e.max():.4f}, mean={e.mean():.4f}"
        )
    treated_ratio = float(df["treatment"].mean())
    print(f"Treated ratio (all): {treated_ratio:.4f}")

    print(
        f"ITE: mean={ite.mean():.6f}, std={ite.std():.6f}, "
        f"min={ite.min():.6f}, max={ite.max():.6f}"
    )
    pct = np.percentile(abs_ite, [10, 25, 50, 75, 90])
    print(
        "Abs(ITE) percentiles (10/25/50/75/90): "
        + ", ".join(f"{p:.6f}" for p in pct)
    )

    if {"y_strict", "y_relaxed", "mu_strict", "mu_relaxed"}.issubset(df.columns):
        for label, cols in [
            ("noisy y", ("y_relaxed", "y_strict")),
            ("clean mu", ("mu_relaxed", "mu_strict")),
        ]:
            a, b = df[cols[0]].mean(), df[cols[1]].mean()
            sa, sb = df[cols[0]].std(), df[cols[1]].std()
            print(f"{label} means (relaxed/strict): {a:.6f}, {b:.6f}")
            print(f"{label} stds  (relaxed/strict): {sa:.6f}, {sb:.6f}")
        y_diff = (df["y_strict"] - df["y_relaxed"]).abs().mean()
        mu_diff = (df["mu_strict"] - df["mu_relaxed"]).abs().mean()
        print(f"Mean |y_strict - y_relaxed|={y_diff:.6f}, |mu_strict - mu_relaxed|={mu_diff:.6f}")

    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
        train_cities = set(train_df["y0_index"].unique())
        test_cities = set(test_df["y0_index"].unique())
        assert train_cities.isdisjoint(test_cities), "Train/test city leakage detected"
        assert not (train_cities & test_cities), "City appears in both splits"

        for name, part in [("train", train_df), ("test", test_df)]:
            n_cities = part["y0_index"].nunique()
            ite_p = part["ite"].to_numpy(dtype=np.float64)
            print(
                f"[{name}] cities={n_cities}, rows={len(part)}, "
                f"treated_ratio={part['treatment'].mean():.4f}, "
                f"ITE mean={ite_p.mean():.6f}, std={ite_p.std():.6f}"
            )
        print("Leakage check: train/test city sets are disjoint.")


# -----------------------------------------------------------------------------
# Causal design notes (brief)
#
# Overlap: deterministic treatment at a geographic level (e.g. all cities in a state get w=0)
# breaks overlap within that stratum; use nontrivial propensity e(X) in (epsilon, 1-epsilon) or
# structured models / extrapolation instead.
#
# Counterfactual separability: beta affects outcomes through transmission; integrate on a daily
# grid (same as the background pool), not only t=[0, H], to avoid rk4 instability and spurious
# zero treatment effects.
# -----------------------------------------------------------------------------


def set_global_seed(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


STATE_NAMES = [
    'Susceptible', 'Exposed', 'Infectious_asymptomatic', 
    'Infectious_pre-symptomatic', 'Infectious_mild', 'Infectious_severe', 
    'Hospitalized_recovered', 'Hospitalized_deceased', 'Recovered', 'Deceased'
]

def get_random_mask_policy():
    return 1 if random.random() < CAUSAL_TREATMENT_ONE_PROB else 0

def load_city_data():
    df = pd.read_csv('sub-est2022.csv')
    # link = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/cities/totals/sub-est2022.csv"
    # df = pd.read_csv(link)
    
    cities_df = df[
        (df['SUMLEV'] == 162) & 
        (df['ESTIMATESBASE2020'] >= 150000) &
        (df['ESTIMATESBASE2020'] < 200000)
    ].sort_values('ESTIMATESBASE2020', ascending=False)
    
    strict_states = [
        'California', 'New York', 'Illinois', 'Washington', 'Oregon',
        'Massachusetts', 'New Jersey', 'Connecticut', 'Rhode Island',
        'Delaware', 'Hawaii', 'Maryland', 'Michigan', 'Nevada',
        'New Mexico', 'Vermont', 'Virginia'
    ]
    
    strict_cities = [
        'New York city', 'Los Angeles city', 'Chicago city', 'San Francisco city',
        'Seattle city', 'Boston city', 'Portland city', 'San Jose city',
        'Denver city', 'Minneapolis city', 'Philadelphia city', 'Sacramento city',
        'San Diego city', 'Oakland city', 'Long Beach city'
    ]
    
    # def get_mask_policy(row):
    #     if row['NAME'] in strict_cities or row['STNAME'] in strict_states:
    #         return 'Strict'
    #     return 'Relaxed'

    def get_mask_policy(row):
        return 'Strict' if random.random() < CAUSAL_TREATMENT_ONE_PROB else 'Relaxed'
    
    cities_df['mask_policy'] = cities_df.apply(get_mask_policy, axis=1)
    
    def assign_parameters(row):
        if row['mask_policy'] == 'Strict':
            delta = 0.15
            alpha = 0.3
        else:
            delta = 0.1
            alpha = 0.5
        return pd.Series({'delta': delta, 'alpha': alpha})
    
    parameter_df = cities_df.apply(assign_parameters, axis=1)
    cities_df = pd.concat([cities_df, parameter_df], axis=1)
    
    output_df = cities_df[['NAME', 'STNAME', 'ESTIMATESBASE2020', 'mask_policy', 'delta', 'alpha']]
    output_df.to_csv('large_cities_with_policies.csv', index=False)
    
    print(f"Total cities analyzed: {len(cities_df)}")
    print("\nMask policy distribution:")
    print(cities_df['mask_policy'].value_counts())
    
    return cities_df


def generate_initial_conditions():
    cities_df = load_city_data()
    populations = cities_df['ESTIMATESBASE2020'].tolist()
    policies = cities_df['mask_policy'].tolist()
    deltas = cities_df['delta'].tolist()
    alphas = cities_df['alpha'].tolist()
    
    initial_conditions = []
    normalized_populations = []
    for pop in populations:
        exposed = int(pop * 0.0015)
        ia = int(pop * 0.001)
        ip = int(pop * 0.0007)
        im = int(pop * 0.0005)
        is_ = int(pop * 0.0002)
        hr = int(pop * 0.00001)
        hd = int(pop * 0.000005)
        r = int(pop * 0.000005)
        d = int(pop * 0.000001)
        s = pop - (exposed + ia + ip + im + is_ + hr + hd + r + d)
        
        y0 = [s, exposed, ia, ip, im, is_, hr, hd, r, d]
        initial_conditions.append(torch.tensor(y0, dtype=torch.float32, device=DEVICE))
        normalized_populations.append(pop)
    
    return initial_conditions, policies, deltas, alphas, normalized_populations



class ODEFunc(nn.Module):
    def __init__(self, population, min_pop, max_pop):
        super(ODEFunc, self).__init__()
        self.N = torch.tensor(float(population), device=DEVICE)
        
        # Constants as buffers (device-safe, not optimized).
        self.register_buffer('Cp', torch.tensor(1.0))
        self.register_buffer('Cm', torch.tensor(1.0))
        self.register_buffer('Cs', torch.tensor(1.0))
        self.register_buffer('mu', torch.tensor(0.928125))

        self.population_factor = 0.8 + (population - min_pop) * (0.2) / (max_pop - min_pop)

        self.Ca = nn.Parameter(torch.tensor(0.425, device=DEVICE))
        self.alpha = nn.Parameter(torch.tensor(0.4875, device=DEVICE))
        self.delta = nn.Parameter(torch.tensor(0.1375, device=DEVICE))
        self.beta = nn.Parameter(torch.tensor(0.5, device=DEVICE))

        # Rate constants (1 / mean duration in days).
        self.gamma = 1/3.5
        self.lambdaa = 1/7
        self.lambdap = 1/1.5
        self.lambdam = 1/5.5
        self.lambdas = 1/5.5
        self.rhor = 1/15
        self.rhod = 1/13.3

    def set_beta(self, beta_val):
        with torch.no_grad():
            self.beta.copy_(torch.tensor(beta_val, dtype=torch.float32, device=DEVICE))

    def forward(self, t, y):
        S, E, Ia, Ip, Im, Is, Hr, Hd, R, D = y

        force_of_infection = self.beta * (self.Ca * Ia + self.Cp * Ip + self.Cm * Im + self.Cs * Is) / self.N
        rate_SE = self.population_factor * S * force_of_infection
        
        rate_EIa = E * self.alpha * self.gamma
        rate_EIp = E * (1 - self.alpha) * self.gamma
        
        rate_IaR = Ia * self.lambdaa
        
        rate_IpIm = Ip * self.mu * self.lambdap
        rate_IpIs = Ip * (1 - self.mu) * self.lambdap
        
        rate_ImR = Im * self.lambdam
        
        rate_IsHr = Is * self.delta * self.lambdas
        rate_IsHd = Is * (1 - self.delta) * self.lambdas
        
        rate_HrR = Hr * self.rhor
        rate_HdD = Hd * self.rhod

        dS = -rate_SE
        dE = rate_SE - rate_EIa - rate_EIp
        dIa = rate_EIa - rate_IaR
        dIp = rate_EIp - rate_IpIs - rate_IpIm
        dIm = rate_IpIm - rate_ImR
        dIs = rate_IpIs - rate_IsHr - rate_IsHd
        dHr = rate_IsHr - rate_HrR
        dHd = rate_IsHd - rate_HdD
        dR = rate_IaR + rate_ImR + rate_HrR
        dD = rate_HdD

        return torch.stack([dS, dE, dIa, dIp, dIm, dIs, dHr, dHd, dR, dD])
    



def generate_inference_csv():
    set_global_seed(RNG_SEED)
    initial_conditions, policies, deltas, alphas, populations = generate_initial_conditions()
    
    min_pop = min(populations)
    max_pop = max(populations)
    
    time_points = 363
    t = torch.arange(0, time_points + 1, dtype=torch.float32, device=DEVICE)
    
    def get_beta_schedule(initial_beta=0.5, lambda_=0.005):
        beta_schedule = []
        current_beta = initial_beta
        
        for _ in range(105):
            beta_schedule.append(current_beta)
        
        for _ in range(time_points + 1 - 105):
            current_beta = current_beta * math.exp(-lambda_)
            beta_schedule.append(current_beta)
        
        return torch.tensor(beta_schedule, dtype=torch.float32, device=DEVICE)
    
    results = []
    
    for i, (y0, policy, delta, alpha) in enumerate(zip(initial_conditions, policies, deltas, alphas)):
        total_population = populations[i]
        
        model = ODEFunc(total_population, min_pop, max_pop).to(DEVICE)
        model.delta = nn.Parameter(torch.tensor(delta, device=DEVICE))
        model.alpha = torch.tensor(alpha, device=DEVICE)
        
        if policy == 'Strict':
            beta_values = get_beta_schedule(initial_beta=0.5, lambda_=0.005)
            
            results.append({
                'Time': t[0].item(),
                'y0_index': i,
                'Population': total_population,
                'beta': beta_values[0].item(),
                'delta': delta,
                'alpha': alpha,
                'Susceptible': y0[0].item() * 10 / total_population,
                'Exposed': y0[1].item() * 10 / total_population,
                'Infectious_asymptomatic': y0[2].item() * 10 / total_population,
                'Infectious_pre-symptomatic': y0[3].item() * 10 / total_population,
                'Infectious_mild': y0[4].item() * 10 / total_population,
                'Infectious_severe': y0[5].item() * 10 / total_population,
                'Hospitalized_recovered': y0[6].item() * 10 / total_population,
                'Hospitalized_deceased': y0[7].item() * 10 / total_population,
                'Recovered': y0[8].item() * 10 / total_population,
                'Deceased': y0[9].item() * 10 / total_population,
            })
            
            for time_step in range(len(beta_values)-1):
                t_half = t[time_step:time_step+2].to(DEVICE)
                model.set_beta(beta_values[time_step].item())
                with torch.no_grad():
                    Y_pred = odeint(model, y0, t_half, method='rk4')
                y0 = Y_pred[-1]
                
                results.append({
                    'Time': t_half[1].item(),
                    'y0_index': i,
                    'Population': total_population,
                    'beta': beta_values[time_step].item(),
                    'delta': delta,
                    'alpha': alpha,
                    'Susceptible': Y_pred[-1, 0].item() * 10 / total_population,
                    'Exposed': Y_pred[-1, 1].item() * 10 / total_population,
                    'Infectious_asymptomatic': Y_pred[-1, 2].item() * 10 / total_population,
                    'Infectious_pre-symptomatic': Y_pred[-1, 3].item() * 10 / total_population,
                    'Infectious_mild': Y_pred[-1, 4].item() * 10 / total_population,
                    'Infectious_severe': Y_pred[-1, 5].item() * 10 / total_population,
                    'Hospitalized_recovered': Y_pred[-1, 6].item() * 10 / total_population,
                    'Hospitalized_deceased': Y_pred[-1, 7].item() * 10 / total_population,
                    'Recovered': Y_pred[-1, 8].item() * 10 / total_population,
                    'Deceased': Y_pred[-1, 9].item() * 10 / total_population,
                })
        
        else:
            model.set_beta(0.5)
            with torch.no_grad():
                Y_pred = odeint(model, y0, t, method='rk4')
            
            for j, t_val in enumerate(t):
                results.append({
                    'Time': t_val.item(),
                    'y0_index': i,
                    'Population': total_population,
                    'beta': 0.5,
                    'delta': delta,
                    'alpha': alpha,
                    'Susceptible': Y_pred[j, 0].item() * 10 / total_population,
                    'Exposed': Y_pred[j, 1].item() * 10 / total_population,
                    'Infectious_asymptomatic': Y_pred[j, 2].item() * 10 / total_population,
                    'Infectious_pre-symptomatic': Y_pred[j, 3].item() * 10 / total_population,
                    'Infectious_mild': Y_pred[j, 4].item() * 10 / total_population,
                    'Infectious_severe': Y_pred[j, 5].item() * 10 / total_population,
                    'Hospitalized_recovered': Y_pred[j, 6].item() * 10 / total_population,
                    'Hospitalized_deceased': Y_pred[j, 7].item() * 10 / total_population,
                    'Recovered': Y_pred[j, 8].item() * 10 / total_population,
                    'Deceased': Y_pred[j, 9].item() * 10 / total_population,
                })

    df = pd.DataFrame(results)
    df.to_csv('Data/daily_10_val_train.csv', index=False)
    return df


def generate_causal_dataset(output_path='covid_causal_14d_dataset.csv', rng_seed: int = RNG_SEED):
    set_global_seed(rng_seed)
    initial_conditions, _, deltas, alphas, populations = generate_initial_conditions()
    
    min_pop = min(populations)
    max_pop = max(populations)
    all_samples = []

    state_names = ['Susceptible', 'Exposed', 'Infectious_asymptomatic', 
                   'Infectious_pre-symptomatic', 'Infectious_mild', 'Infectious_severe', 
                   'Hospitalized_recovered', 'Hospitalized_deceased', 'Recovered', 'Deceased']

    print("Generating samples...")

    for i in range(len(populations)):
        pop = populations[i]
        y0_init = initial_conditions[i]

        model = ODEFunc(pop, min_pop, max_pop).to(DEVICE)
        # model.delta = nn.Parameter(torch.tensor(deltas[i], device=DEVICE))
        # model.alpha = torch.tensor(alphas[i], device=DEVICE)

        model.alpha = nn.Parameter(torch.tensor(alphas[i], dtype=torch.float32, device=DEVICE))
        model.delta = nn.Parameter(torch.tensor(deltas[i], dtype=torch.float32, device=DEVICE))

        # Background trajectory: 180 days at beta=0.5
        model.set_beta(0.5)
        t_pool = torch.arange(0, 180, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pool_data = odeint(model, y0_init, t_pool, method='rk4')

        for start_t in range(0, 150, 7):
            hist_end = start_t + 14

            X_hist = pool_data[start_t:hist_end].cpu().numpy()
            y_anchor = pool_data[hist_end - 1].detach().clone()

            e = treatment_propensity_from_window(X_hist, float(pop))
            a_factual = 1 if random.random() < e else 0

            t_future = torch.arange(
                0.0, float(CAUSAL_FUTURE_DAYS) + 1.0, dtype=torch.float32, device=DEVICE
            )

            beta_s, beta_r = beta_for_arms_from_window(X_hist, y_anchor, float(pop))

            model.set_beta(beta_s)
            with torch.no_grad():
                y_s = odeint(model, y_anchor.clone(), t_future, method='rk4')[-1]
            model.set_beta(beta_r)
            with torch.no_grad():
                y_r = odeint(model, y_anchor.clone(), t_future, method='rk4')[-1]

            def death_increment_normalized(y_end: torch.Tensor) -> float:
                return max(0.0, (y_end[9] - y_anchor[9]).item() * 10 / pop)

            mu_strict = death_increment_normalized(y_s)
            mu_relaxed = death_increment_normalized(y_r)

            y_strict = add_outcome_noise(mu_strict, X_hist, float(pop), arm=1)
            y_relaxed = add_outcome_noise(mu_relaxed, X_hist, float(pop), arm=0)

            if a_factual == 1:
                outcome_f = y_strict
                outcome_cf = y_relaxed
            else:
                outcome_f = y_relaxed
                outcome_cf = y_strict

            ite = mu_strict - mu_relaxed
            ite_rel_minus_strict = mu_relaxed - mu_strict

            sample_data = {
                'sample_id': f"city_{i}_t{start_t}",
                'y0_index': i,
                'start_time': start_t,
                'treatment': a_factual,
                'propensity': float(e),
                'y_factual': float(outcome_f),
                'y_counterfactual': float(outcome_cf),
                'y_strict': float(y_strict),
                'y_relaxed': float(y_relaxed),
                'mu_strict': float(mu_strict),
                'mu_relaxed': float(mu_relaxed),
                'ite': float(ite),
                'ite_rel_minus_strict': float(ite_rel_minus_strict),
                'pop_size': int(pop)
            }
            
            for d in range(14):
                for s_idx, s_name in enumerate(state_names):
                    sample_data[f"x_d{d}_{s_name}"] = float(X_hist[d, s_idx] * 10 / pop)
            
            all_samples.append(sample_data)

    df = pd.DataFrame(all_samples)

    city_split = assign_city_splits(df["y0_index"].tolist())
    df["split"] = df["y0_index"].map(city_split)

    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {len(df)} samples -> {output_path}")
    _print_causal_diagnostics(df)
    return df


# df = generate_inference_csv()
# df.head()

if __name__ == '__main__':
    set_global_seed(RNG_SEED)
    df = generate_causal_dataset()
    df.head()