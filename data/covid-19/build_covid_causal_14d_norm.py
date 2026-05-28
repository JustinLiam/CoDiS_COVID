"""
Build CoDiS-ready norm_data and mask CSVs from covid_causal_14d_dataset.csv.

Column layout matches ACIC: t, y_0, y_1, mu_0, mu_1, X (standardized).
StandardScaler fits one scalar mean/std per *column*; large pop_size only affects
that column's scale, not other features.

Usage (from repo root):
  python ./build_covid_causal_14d_norm.py
  python ./build_covid_causal_14d_norm.py --split train --out-id covid_causal_14d_train
  python ./build_covid_causal_14d_norm.py --split test  --out-id covid_causal_14d_test
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
    "propensity",
    "y_strict",
    "y_relaxed",
    "mu_strict",
    "mu_relaxed",
    "split",
}


def build_potential_outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: reconstruct y_0/y_1 from factual/counterfactual when y_strict/y_relaxed absent."""
    t = df["treatment"].to_numpy()
    yf = df["y_factual"].to_numpy(dtype=np.float64)
    ycf = df["y_counterfactual"].to_numpy(dtype=np.float64)
    y0 = np.where(t == 0, yf, ycf)
    y1 = np.where(t == 1, yf, ycf)
    return y0, y1


def build_noisy_potential_outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """y_0 = noisy relaxed, y_1 = noisy strict (preferred when columns exist)."""
    if {"y_strict", "y_relaxed"}.issubset(df.columns):
        return (
            df["y_relaxed"].to_numpy(dtype=np.float64),
            df["y_strict"].to_numpy(dtype=np.float64),
        )
    return build_potential_outcomes(df)


def build_clean_potential_outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """mu_0 = clean relaxed, mu_1 = clean strict; fallback to noisy y for old CSVs."""
    if {"mu_strict", "mu_relaxed"}.issubset(df.columns):
        return (
            df["mu_relaxed"].to_numpy(dtype=np.float64),
            df["mu_strict"].to_numpy(dtype=np.float64),
        )
    return build_potential_outcomes(df)


def print_split_diagnostics(raw: pd.DataFrame, sub: pd.DataFrame, y0: np.ndarray, y1: np.ndarray,
                            mu0: np.ndarray, mu1: np.ndarray, split_label: str) -> None:
    ite = mu1 - mu0
    print(f"\n=== Norm build diagnostics ({split_label}) ===")
    print(f"Rows: {len(sub)}")
    if "y0_index" in sub.columns:
        print(f"Cities: {sub['y0_index'].nunique()}")
    print(f"Treated ratio: {sub['treatment'].mean():.4f}")
    print(f"ITE mean={ite.mean():.6f}, std={ite.std():.6f}")
    print(f"y_0 mean={y0.mean():.6f}, std={y0.std():.6f}")
    print(f"y_1 mean={y1.mean():.6f}, std={y1.std():.6f}")
    print(f"mu_0 mean={mu0.mean():.6f}, std={mu0.std():.6f}")
    print(f"mu_1 mean={mu1.mean():.6f}, std={mu1.std():.6f}")
    y_mu_diff = np.mean(np.abs(y0 - mu0) + np.abs(y1 - mu1))
    print(f"Mean |y - mu| (both arms): {y_mu_diff:.6f}")

    if "split" in raw.columns and split_label == "all":
        for part_name in ("train", "test"):
            part = raw[raw["split"] == part_name]
            if len(part) == 0:
                continue
            train_cities = set(raw.loc[raw["split"] == "train", "y0_index"].unique())
            test_cities = set(raw.loc[raw["split"] == "test", "y0_index"].unique())
            assert train_cities.isdisjoint(test_cities)
            mu_s = part["mu_strict"].to_numpy(dtype=np.float64) if "mu_strict" in part.columns else None
            mu_r = part["mu_relaxed"].to_numpy(dtype=np.float64) if "mu_relaxed" in part.columns else None
            if mu_s is not None and mu_r is not None:
                ite_part = mu_s - mu_r
                print(
                    f"[{part_name}] cities={part['y0_index'].nunique()}, rows={len(part)}, "
                    f"treated_ratio={part['treatment'].mean():.4f}, "
                    f"ITE mean={ite_part.mean():.6f}, std={ite_part.std():.6f}"
                )


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
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Filter raw rows by city-level split column (requires split in CSV)",
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

    if args.split != "all":
        if "split" not in raw.columns:
            print(
                f"ERROR: --split {args.split} requested but 'split' column missing in {args.input}",
                file=sys.stderr,
            )
            sys.exit(1)
        raw = raw[raw["split"] == args.split].reset_index(drop=True)
        if len(raw) == 0:
            print(f"ERROR: no rows for split={args.split}", file=sys.stderr)
            sys.exit(1)

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
        raw = raw.loc[~bad].reset_index(drop=True)

    y0, y1 = build_noisy_potential_outcomes(raw)
    mu0, mu1 = build_clean_potential_outcomes(raw)

    print_split_diagnostics(raw, raw, y0, y1, mu0, mu1, args.split)

    t_col = sub["treatment"].to_numpy(dtype=np.float64).reshape(-1, 1)
    x_table = sub[x_cols].to_numpy(dtype=np.float64)

    cov_scalar = preprocessing.StandardScaler()
    x = cov_scalar.fit_transform(x_table)

    y_out_scaler = preprocessing.StandardScaler()
    y_out_scaler.fit(np.concatenate([y0, y1]).reshape(-1, 1))
    y_0 = y_out_scaler.transform(y0.reshape(-1, 1))
    y_1 = y_out_scaler.transform(y1.reshape(-1, 1))

    mu_out_scaler = preprocessing.StandardScaler()
    mu_out_scaler.fit(np.concatenate([mu0, mu1]).reshape(-1, 1))
    mu_0 = mu_out_scaler.transform(mu0.reshape(-1, 1))
    mu_1 = mu_out_scaler.transform(mu1.reshape(-1, 1))

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
    n_x_day = sum(1 for c in x_cols if c.startswith("x_d"))
    print(f"x_dim (covariates) = {len(x_cols)}, eval_length = {full.shape[1]}")
    if n_x_day > 0:
        n_days = max(int(c.split("_")[1][1:]) for c in x_cols if c.startswith("x_d")) + 1
        print(f"Historical window days in X: {n_days} (expected 14)")


if __name__ == "__main__":
    main()
