"""
Build CoDiS-ready norm_data and mask CSVs from covid_causal_14d_dataset.csv.

Column layout matches ACIC: t, y_0, y_1, mu_0, mu_1, X (standardized).
StandardScaler fits one scalar mean/std per *column*; large pop_size only affects
that column's scale, not other features.

Usage (from repo root):
  python data/covid-19/build_covid_causal_14d_norm.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

EXCLUDE_FROM_X = {
    "sample_id",
    "y0_index",
    "start_time",
    "treatment",
    "y_factual",
    "y_counterfactual",
    "ite",
    "ite_rel_minus_strict",
}


def build_potential_outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df["treatment"].to_numpy()
    yf = df["y_factual"].to_numpy(dtype=np.float64)
    ycf = df["y_counterfactual"].to_numpy(dtype=np.float64)
    y0 = np.where(t == 0, yf, ycf)
    y1 = np.where(t == 1, yf, ycf)
    return y0, y1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(__file__), "covid_causal_14d_dataset.csv"
        ),
        help="Source CSV path",
    )
    parser.add_argument(
        "--out-id",
        default="covid_causal_14d",
        help="Output basename (without .csv) for norm_data and mask",
    )
    parser.add_argument(
        "--norm-dir",
        default=os.path.join(os.path.dirname(__file__), "covid_norm_data"),
    )
    parser.add_argument(
        "--mask-dir",
        default=os.path.join(os.path.dirname(__file__), "covid_mask"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    raw.replace("?", np.nan, inplace=True)

    for c in raw.columns:
        if c in EXCLUDE_FROM_X or c == "sample_id":
            continue
        if raw[c].dtype == object:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    x_cols = [c for c in raw.columns if c not in EXCLUDE_FROM_X and c != "sample_id"]
    needed = ["treatment", "y_factual", "y_counterfactual"] + x_cols
    sub = raw[needed].copy()
    bad = sub.isnull().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"Dropping {n_bad} rows with missing values.", file=sys.stderr)
        sub = sub.loc[~bad].reset_index(drop=True)

    y0, y1 = build_potential_outcomes(
        pd.DataFrame(
            {
                "treatment": sub["treatment"],
                "y_factual": sub["y_factual"],
                "y_counterfactual": sub["y_counterfactual"],
            }
        )
    )
    t_col = sub["treatment"].to_numpy(dtype=np.float64).reshape(-1, 1)
    x_table = sub[x_cols].to_numpy(dtype=np.float64)

    cov_scalar = preprocessing.StandardScaler()
    x = cov_scalar.fit_transform(x_table)

    y_out_scaler = preprocessing.StandardScaler()
    y_out_scaler.fit(np.concatenate([y0, y1]).reshape(-1, 1))
    y_0 = y_out_scaler.transform(y0.reshape(-1, 1))
    y_1 = y_out_scaler.transform(y1.reshape(-1, 1))

    mu_out_scaler = preprocessing.StandardScaler()
    mu_out_scaler.fit(np.concatenate([y0, y1]).reshape(-1, 1))
    mu_0 = mu_out_scaler.transform(y0.reshape(-1, 1))
    mu_1 = mu_out_scaler.transform(y1.reshape(-1, 1))

    full = np.concatenate([t_col, y_0, y_1, mu_0, mu_1, x], axis=1)

    lead_names = ["t", "y_0", "y_1", "mu_0", "mu_1"]
    col_names = lead_names + list(x_cols)

    os.makedirs(args.norm_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    norm_path = os.path.join(args.norm_dir, args.out_id + ".csv")
    mask_path = os.path.join(args.mask_dir, args.out_id + ".csv")

    norm_df = pd.DataFrame(full, columns=col_names)
    norm_df.to_csv(norm_path, index=False)
    print(f"Wrote {norm_path} shape={full.shape}")

    mask = np.ones(full.shape, dtype=np.float64)
    mask[:, 3] = 0.0
    mask[:, 4] = 0.0
    for i in range(full.shape[0]):
        tt = full[i, 0]
        if tt == 0:
            mask[i, 2] = 0.0
        if tt == 1:
            mask[i, 1] = 0.0

    mask_names = ["mask_" + n for n in col_names]
    pd.DataFrame(mask, columns=mask_names).to_csv(mask_path, index=False)
    print(f"Wrote {mask_path} shape={mask.shape}")

    assert full.shape[1] == len(col_names) == 5 + len(x_cols)
    print(f"x_dim (covariates) = {len(x_cols)}, eval_length = {full.shape[1]}")


if __name__ == "__main__":
    main()
