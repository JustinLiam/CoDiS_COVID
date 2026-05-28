# CoDiS

Implementation of **Conditional Diffusion for Causal Inference with State Space Representation**.

---

## Introduction

Causal inference from observational data, particularly for estimating individual-level effects, is vital in high-stakes domains such as healthcare and economics. A key difficulty lies in modeling complex dependencies, such as structurally coupled and temporal dynamics among observational covariates, which are often overlooked by existing methods. Furthermore, most approaches only provide point estimates for potential outcomes, neglecting the distributional information needed to quantify heterogeneity and fine-grained disparities among individuals.To address these gaps, we propose CoDiS, a conditional diffusion framework for estimating potential outcome distributions through a structure-aware state space denoising architecture. Specifically, CoDiS jointly processes the covariate-treatment condition, the noisy outcome state, and diffusion timestep with an SSM-based backbone during reverse denoising. This design allows the reverse denoising process to exploit latent dependencies among observational variables, thereby improving the potential outcome distribution generation. Extensive evaluations on a COVID-19 dataset and standard causal benchmarks demonstrate the effectiveness of CoDiS in both distributional and point estimation.

---

## Setup

### Environment

- **Python** 3.8+ (developed with 3.8.18)
- **PyTorch** 1.12.1 with CUDA (recommended for training; the SSM backbone uses a CUDA Cauchy extension when available)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Optional but recommended for faster SSM training (10–50× speedup on GPU):

```bash
cd src/entensions/cauchy && python setup.py install && cd ../../..
```

### Weights & Biases (optional)

Training scripts integrate **[Weights & Biases](https://wandb.ai/)** for logging (`train loss`, validation PEHE, etc.).  
**By default, `exe_covid.py` uses `--wandb_mode disabled`**, so runs work without a W&B account.

To enable logging:

```bash
wandb login   # once, for online mode
python exe_covid.py --wandb_mode online   # or offline
```

---

## Data layout

Place datasets under `data/`. The **primary walkthrough below uses COVID-19**; other folders support additional benchmarks from the paper.

```text
data/
├── covid-19/                 # COVID semi-synthetic 14-day benchmark (recommended entry point)
│   ├── README.md             # schema, checksums, regeneration — read this for COVID details
│   ├── covid_norm_data/      # standardized tables for CoDiS
│   └── covid_mask/           # observation masks (ACIC-style)
├── acic2018/                 # ACIC 2018 (exe_acic.py, config/acic2018.yaml)
├── acic2016/                 # ACIC 2016
├── ihdp/                     # IHDP
└── synthetic_data/           # optional synthetic benchmarks (not covered in this README)
```

### COVID-19 benchmark (summary)

Semi-synthetic causal data from a compartment ODE simulator over US cities: **14-day** covariate trajectories, binary treatment (strict vs relaxed policy), and noisy/clean potential outcomes in an ACIC-compatible table layout.

| Item | Value |
|------|--------|
| Default ID | `covid_causal_14d` |
| Pre-split train / test rows | **1034 / 264** |
| Covariate dimension | **141** (`pop_size` + 14×10 SEIRD states) |
| CoDiS inputs | `x_seq` shape **(11, 14)** — population + 10 dynamic compartments over 14 days |

**Obtain the files:** use the pre-built CSVs under `data/covid-19/covid_norm_data/` and `covid_mask/`, or regenerate from scratch — see [`data/covid-19/README.md`](data/covid-19/README.md) for checksums, column schema, and build scripts (`simulate_daily_data_train.py`, `build_covid_causal_14d_norm.py`).

**Train / valid / test inside `exe_covid.py`:** controlled by `dataset.split` in `config/covid.yaml`.

- **`split: "files"`** (default): train on `covid_causal_14d_train.csv`; hold out **~10%** of those rows for validation; test on `covid_causal_14d_test.csv` only (**931 / 103 / 264** samples for the default ID and seed).
- **`split: "random"`**: single table `covid_causal_14d.csv` with random **~72% / ~8% / ~20%** train / valid / test row splits.

PropensityNet is fit on **`_train.csv` only** when `split: "files"` (no test leakage).

---

## Quick start (COVID-19)

Hyperparameters live in [`config/covid.yaml`](config/covid.yaml). Entry script: [`exe_covid.py`](exe_covid.py).

**Smoke test** (dataloader + propensity net + one forward pass):

```bash
python exe_covid.py --config covid.yaml --current_id covid_causal_14d --smoke_test 1 --train 0
```

**Train and evaluate** (default W&B disabled):

```bash
CUDA_VISIBLE_DEVICES=0 python exe_covid.py \
  --config covid.yaml \
  --current_id covid_causal_14d \
  --seed 20
```

**Multiple seeds** (example):

```bash
for seed in 20 202 2020 20202 202020; do
  CUDA_VISIBLE_DEVICES=0 python exe_covid.py \
    --config covid.yaml \
    --current_id covid_causal_14d \
    --seed "$seed"
done
```

Outputs:

- Run configs and logs: `./save_diffs4/covid/`
- Model weights: `./save_model/covid/covid_causal_14d/model_weights.pth`

**COVID-19 density figures** (optional, requires `seaborn`):

```bash
python plot_covid19_densities.py --config covid.yaml --current_id covid_causal_14d
```

---

## Other benchmarks (ACIC / IHDP)

These follow the same CoDiS training loop with dataset-specific loaders and configs:

| Dataset | Runner | Config | Notes |
|---------|--------|--------|-------|
| ACIC 2018 | `exe_acic.py` | `config/acic2018.yaml` | Preprocess with `load_acic2018.ipynb`; example: `script/script_acic2018.sh` |

Additional public benchmarks (IHDP, ACIC 2016) can be placed under `data/ihdp/` and `data/acic2016/` using the same `{dataset}_norm_data/` + `{dataset}_mask/` layout; see `PropensityNet.py` and `dataset_acic.py` for path conventions.

Download links: [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition), [ACIC 2018](https://www.synapse.org/Synapse:syn11294478/wiki/486304), [IHDP (CEVAE)](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets).

---

## Repository map (minimal)

| Path | Role |
|------|------|
| `exe_covid.py` | COVID training / evaluation entry point |
| `dataset_covid.py` | COVID dataloading and train–valid–test splits |
| `PropensityNet.py` | Propensity score network (IPW) |
| `src/main_model.py`, `src/diff_model.py` | CoDiS diffusion model (SSM + Transformer backbone) |
| `config/covid.yaml` | Default COVID hyperparameters |
| `data/covid-19/README.md` | Full COVID dataset documentation |
