# COVID-19 Semi-Synthetic Causal Dataset (14-day window)

Semi-synthetic causal benchmark built from a compartment ODE simulator over 59 US cities.
Designed for treatment effect estimation methods including **CoDiS** and **CATENets**.

Each sample uses a **14-day** epidemiological history as covariates `X`, a **30-day** forward
simulation horizon for outcomes, and binary treatment:

| Code | Policy   |
|------|----------|
| `1`  | Strict   |
| `0`  | Relaxed  |

---

## Version pin (verify your download)

| Field | Value |
|-------|-------|
| **Generated on** | 2026-05-21 |
| **Source git commit** | `572ad38` (CATENets-main at generation time; update after publishing this repo) |
| **Random seed** | `42` (`RNG_SEED`, `CAUSAL_SPLIT_SEED`) |
| **Cities** | 59 total → **47 train** / **12 test** (city-level split, no sliding-window leakage) |
| **Raw samples** | 1298 |
| **Norm train / test rows** | **1034 / 264** |
| **Covariate dim** | 141 (`pop_size` + 14×10 SEIRD states) |
| **Norm column count** | 146 (`t`, `y_0`, `y_1`, `mu_0`, `mu_1`, X…) |

### SHA-256 checksums

```text
2b0241e050a1161efd8a1eeced624d4b2cf9536f5933936c9cff7f4e24cc37a0  covid_causal_14d_dataset.csv
c7ed50b0288557fb439a3ec7c87a0ebffa2970198a72d9b13147f8ad3fe7e395  covid_norm_data/covid_causal_14d_train.csv
edf25b5837785d07871e11e64a7e0c3be0b8dbf76f4cb0b2f3b89cf3095b16d2  covid_norm_data/covid_causal_14d_test.csv
00612e62104ebcce6f49b21ad9ae6f272f84364a1453ae91914858929416d3a7  covid_mask/covid_causal_14d_train.csv
a9e980d9246ff641868e1fef9e5d956dd30f5d2c4f2570d655ac3ac5f19203c8  covid_mask/covid_causal_14d_test.csv
```

Quick row check:

```bash
wc -l covid_causal_14d_dataset.csv covid_norm_data/covid_causal_14d_train.csv covid_norm_data/covid_causal_14d_test.csv
# Expected: 1299, 1035, 265  (includes header line each)
```

---

## Directory layout

```text
.
├── README.md
├── requirements-data.txt
├── simulate_daily_data_train.py      # Step 1: ODE simulation → raw CSV
├── build_covid_causal_14d_norm.py    # Step 2: raw → norm + mask CSVs
├── covid_causal_14d_dataset.csv      # Raw semi-synthetic dataset (pre-generated)
├── covid_norm_data/
│   ├── covid_causal_14d_train.csv    # CoDiS / CATENets training split
│   └── covid_causal_14d_test.csv     # Held-out city-level test split
└── covid_mask/
    ├── covid_causal_14d_train.csv    # Observation mask (ACIC-style)
    └── covid_causal_14d_test.csv
```

**Downstream methods should use `covid_norm_data/*_{train,test}.csv`** (and matching `covid_mask/` if your pipeline needs masks). Do not re-standardize features; scalers were fit at build time.

---

## Raw CSV schema (`covid_causal_14d_dataset.csv`)

Key columns (full list includes flattened `x_d{d}_{state}` for d=0..13):

| Column | Description |
|--------|-------------|
| `sample_id`, `y0_index`, `start_time` | Sample / city / window start (excluded from X) |
| `treatment` | Factual treatment (1=strict, 0=relaxed) |
| `propensity` | Confounded assignment score ∈ [0.15, 0.85] |
| `y_factual`, `y_counterfactual` | Noisy factual / counterfactual outcomes |
| `y_strict`, `y_relaxed` | Noisy potential outcomes |
| `mu_strict`, `mu_relaxed` | Clean ODE potential outcomes |
| `ite` | `mu_strict - mu_relaxed` |
| `split` | `train` or `test` (city-level) |
| `pop_size` | City population |
| `x_d{d}_*` | 14-day × 10-state trajectory (per-capita) |

---

## Normalized CSV schema (`covid_norm_data/*.csv`)

ACIC-compatible single-table layout (already standardized):

```text
t, y_0, y_1, mu_0, mu_1, pop_size, x_d0_Susceptible, …, x_d13_Deceased
```

| Column | Meaning |
|--------|---------|
| `t` | Treatment |
| `y_0`, `y_1` | Noisy potential outcomes (relaxed / strict), scaled |
| `mu_0`, `mu_1` | Clean potential outcomes (relaxed / strict), scaled |
| Remaining columns | Standardized covariates |

Mask files mirror column names with `mask_` prefix; unobserved potential outcomes are masked per factual arm.

---

## Regenerate from scratch

### 1. Dependencies

```bash
pip install -r requirements-data.txt
```

### 2. Census city populations (not included in this repo)

Download and place **`sub-est2022.csv`** in the same directory as `simulate_daily_data_train.py`:

https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/cities/totals/sub-est2022.csv

The script reads the local file by default (`pd.read_csv('sub-est2022.csv')`). We **do not** fetch from URL at runtime so runs stay reproducible and work offline. City filter: incorporated places (`SUMLEV=162`) with 2020 population in **[150,000, 200,000)** → 59 cities.

### 3. Generate raw dataset (~1–2 min CPU/GPU)

```bash
cd /path/to/this/folder
python simulate_daily_data_train.py
# → covid_causal_14d_dataset.csv
```

### 4. Build norm + mask splits

```bash
python build_covid_causal_14d_norm.py --split train --out-id covid_causal_14d_train
python build_covid_causal_14d_norm.py --split test  --out-id covid_causal_14d_test
```

Outputs land in `covid_norm_data/` and `covid_mask/` next to the build script.

---

## Usage with CoDiS

Point your CoDiS data config to:

- **Train:** `covid_norm_data/covid_causal_14d_train.csv`
- **Test:** `covid_norm_data/covid_causal_14d_test.csv`
- **Masks (if required):** matching files under `covid_mask/`

Input dimensionality: **141** covariates + lead columns as defined by your loader. Historical length: **14 days** × **10** compartments.

---

## Usage with CATENets

Example loader paths (relative to `catenets/datasets/covid-19/`):

```python
from catenets.datasets import load

X, w, y, po_train, X_test, w_test, y_test, po_test = load(
    "covid19",
    dataset_index="covid_norm_data/covid_causal_14d_train.csv",
    # test file inferred automatically from *_train.csv → *_test.csv
)
```

---

## Citation

If you use this dataset, cite the **CoDiS** paper and mention this semi-synthetic COVID-19 benchmark. Update with your BibTeX once published.

---

## License / third-party data

- **Simulation code and synthetic outputs:** follow the license of the repository you publish from.
- **`sub-est2022.csv`:** U.S. Census Bureau public data; obtain separately from the link above.
