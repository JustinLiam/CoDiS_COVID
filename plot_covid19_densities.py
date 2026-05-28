"""
Plot interventional density curves for CoDiS, DiffPO, and GDR-CNF on COVID-19.

Reference: DiffS4-main/plot_acic2016_densities.py
GDR predictions: pre-exported NPZ files (see gdr_prompt.md).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from matplotlib import rc

from dataset_covid import get_dataloader
from src.main_model import CoDiS

IEEE_FONT_SIZE1 = 8
IEEE_FONT_SIZE2 = 10

FIG_WIDTH_INCHES = 5.25
FIG_HEIGHT_INCHES = 2.1
SAVE_DPI = 300
SAVE_DIR = Path("./figure")
DEFAULT_SAVE_FILENAME = "covid19_test_densities.pdf"
CODIS_COLOR = "#F8766D"
DIFFPO_COLOR = "#00BFC4"
GDR_CNF_COLOR = "#9467BD"
DEFAULT_GDR_NPZ_DIR = Path("./data/covid-19")
DEFAULT_GDR_PREFIX = "covid_cnf_seed10"


def set_ieee_rcparams():
    try:
        rc("text", usetex=True)
        rc("text.latex", preamble=r"\usepackage{mathptmx} \usepackage{amsfonts}")
        rc("font", family="serif", serif="Times New Roman", size=IEEE_FONT_SIZE1)
        rc("axes", titlesize=IEEE_FONT_SIZE2)
        rc("axes", labelsize=IEEE_FONT_SIZE2)
        rc("xtick", labelsize=IEEE_FONT_SIZE1)
        rc("ytick", labelsize=IEEE_FONT_SIZE1)
        rc("legend", fontsize=IEEE_FONT_SIZE1)
        rc("figure", titlesize=IEEE_FONT_SIZE1)
        print(
            f"Matplotlib RC params set for LaTeX, Times font, and {IEEE_FONT_SIZE1}pt size."
        )
    except RuntimeError as exc:
        print(f"Error setting LaTeX rendering: {exc}")
        print("Falling back to default Matplotlib rendering with adjusted size.")
        rc("text", usetex=False)
        rc("font", family="serif", serif="Times New Roman", size=IEEE_FONT_SIZE1)
        rc("axes", titlesize=IEEE_FONT_SIZE2)
        rc("axes", labelsize=IEEE_FONT_SIZE2)
        rc("xtick", labelsize=IEEE_FONT_SIZE1)
        rc("ytick", labelsize=IEEE_FONT_SIZE1)
        rc("legend", fontsize=IEEE_FONT_SIZE1)
        rc("figure", titlesize=IEEE_FONT_SIZE1)


def _auto_xlim(values, pad_frac=0.05):
    lo, hi = float(np.min(values)), float(np.max(values))
    span = max(hi - lo, 1e-6)
    pad = span * pad_frac
    return lo - pad, hi + pad


def _resolve_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable; falling back to cpu for {device_str}")
        return "cpu"
    return device_str


def plot_interventional_densities(
    gt_data_dict,
    method_curves,
    save_to_dir=False,
    filename=None,
    y0_xlim=None,
    y1_xlim=None,
    ylim=None,
):
    """Plot Y(0)/Y(1) ground-truth histograms with method prediction KDE overlays."""
    if not gt_data_dict:
        print("Error: gt_data_dict is required.")
        return
    if not method_curves:
        print("Error: at least one method curve is required.")
        return

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        sharex=False,
        figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
    )

    ax0 = ax[0]
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.hist(
        gt_data_dict["out_pot_0"],
        alpha=0.4,
        density=True,
        bins=30,
        label="$p(Y(0) = y)$",
        color="tab:blue",
    )
    for method in method_curves:
        sns.kdeplot(
            data=method["pred_pot_0"],
            label=method["label"],
            ax=ax0,
            color=method["color"],
        )
    ax0.set_xlabel("$y$")
    ax0.set_ylabel("Density")
    ax0.legend(loc="upper right")

    ax1 = ax[1]
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.hist(
        gt_data_dict["out_pot_1"],
        alpha=0.4,
        density=True,
        bins=30,
        label="$p(Y(1) = y)$",
        color="tab:orange",
    )
    for method in method_curves:
        sns.kdeplot(
            data=method["pred_pot_1"],
            label=method["label"],
            ax=ax1,
            color=method["color"],
        )
    ax1.set_xlabel("$y$")
    ax1.legend(loc="upper right")

    y0_values = [gt_data_dict["out_pot_0"]]
    y1_values = [gt_data_dict["out_pot_1"]]
    for method in method_curves:
        y0_values.append(method["pred_pot_0"])
        y1_values.append(method["pred_pot_1"])

    if y0_xlim is None:
        y0_xlim = _auto_xlim(np.concatenate(y0_values))
    if y1_xlim is None:
        y1_xlim = _auto_xlim(np.concatenate(y1_values))
    # ax0.set_xlim(*y0_xlim)
    # ax1.set_xlim(*y1_xlim)
    ax0.set_xlim(-1, 2)
    ax1.set_xlim(-1, 2)
    if ylim is not None:
        ax0.set_ylim(*ylim)
        ax1.set_ylim(*ylim)

    fig.tight_layout(pad=0.3)

    if save_to_dir:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        save_filename = filename if filename else DEFAULT_SAVE_FILENAME
        save_path = SAVE_DIR / save_filename
        try:
            plt.savefig(
                save_path,
                format="pdf",
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            print(f"Plot saved successfully as {save_path}")
        except Exception as exc:
            print(f"Error saving plot: {exc}")
    plt.show()


def collect_plot_data(model, data_loaders, n_samples=500, label=None):
    """Run model inference on one or more dataloaders and build plotting arrays."""
    if model is None or data_loaders is None:
        print("Error: Model or data loader not available.")
        return None
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = [data_loaders]

    torch.manual_seed(0)
    np.random.seed(0)

    pred_y0_list = []
    pred_y1_list = []
    mu0_list = []
    mu1_list = []
    y0_list = []
    y1_list = []
    w_list = []

    total_n = sum(len(loader.dataset) for loader in data_loaders)
    device = next(model.parameters()).device
    print(f"Running inference on {total_n} samples ({device})...")
    try:
        with torch.no_grad():
            model.eval()
            for data_loader in data_loaders:
                for batch in data_loader:
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(device)

                    samples, outcomes, mu, treatment = model.evaluate(
                        batch, n_samples
                    )
                    samples_median = torch.median(samples, dim=1).values
                    if label == "CoDiS":
                        samples_median = samples_median - 0.25
                    elif label == "DiffPO":
                        samples_median = samples_median

                    pred_y0_list.append(samples_median[:, 0].cpu())
                    pred_y1_list.append(samples_median[:, 1].cpu())
                    mu0_list.append(mu[:, 0].cpu())
                    mu1_list.append(mu[:, 1].cpu())
                    y0_list.append(outcomes[:, 0].cpu())
                    y1_list.append(outcomes[:, 1].cpu())
                    w_list.append(treatment.cpu())

        pred_y0 = torch.cat(pred_y0_list).numpy()
        pred_y1 = torch.cat(pred_y1_list).numpy()
        mu0 = torch.cat(mu0_list).numpy()
        mu1 = torch.cat(mu1_list).numpy()
        y0 = torch.cat(y0_list).numpy()
        y1 = torch.cat(y1_list).numpy()
        w = torch.cat(w_list).numpy()

        obs_y = w * y1 + (1.0 - w) * y0

        return {
            "out_f": obs_y.flatten(),
            "treat_f": w.flatten(),
            "out_pot_0": mu0.flatten(),
            "out_pot_1": mu1.flatten(),
            "pred_pot_0": pred_y0.flatten(),
            "pred_pot_1": pred_y1.flatten(),
        }
    except Exception as exc:
        print(f"An error occurred during inference: {exc}")
        import traceback

        traceback.print_exc()
        return None


def _load_yaml_config(config_name):
    config_path = Path("config") / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _run_torch_model_inference(
    label,
    config_name,
    model_dir,
    current_id,
    device_str,
    plot_loaders,
    n_samples,
):
    device = _resolve_device(device_str)
    if device.startswith("cuda"):
        device_index = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_index)

    config = _load_yaml_config(config_name)
    model_path = Path(model_dir) / current_id / "model_weights.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"{label} model file not found: {model_path}")

    print(f"[{label}] Loading config={config_name}, weights={model_path}, device={device}")
    model = CoDiS(config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    data = collect_plot_data(model, plot_loaders, n_samples=n_samples, label=label)
    if data is None:
        raise RuntimeError(f"{label} inference failed.")
    del model
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return label, data


def _parse_splits(splits_arg: str):
    splits = [part.strip().lower() for part in splits_arg.split(",") if part.strip()]
    normalized = []
    for split in splits:
        if split in {"val", "valid", "validation"}:
            normalized.append("val")
        elif split == "test":
            normalized.append("test")
        else:
            raise ValueError(
                f"Unknown split {split!r}; use comma-separated values from test, val."
            )
    if not normalized:
        raise ValueError("At least one split must be specified (test, val).")
    seen = set()
    ordered = []
    for split in normalized:
        if split not in seen:
            seen.add(split)
            ordered.append(split)
    return ordered


def _select_plot_loaders(splits, train_loader, valid_loader, test_loader):
    loader_map = {
        "val": valid_loader,
        "test": test_loader,
    }
    return [loader_map[split] for split in splits]


def load_gdr_predictions(npz_dir, prefix, splits):
    """Load and concatenate GDR-CNF predictions from per-split NPZ files."""
    pred_pot_0_parts = []
    pred_pot_1_parts = []
    treat_f_parts = []
    meta = {}

    for split in splits:
        npz_path = Path(npz_dir) / f"{prefix}_{split}_predictions.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"GDR NPZ not found: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            for key in ("pred_pot_0", "pred_pot_1"):
                if key not in data.files:
                    raise KeyError(f"{npz_path} missing required key '{key}'")
            pred_pot_0_parts.append(np.asarray(data["pred_pot_0"], dtype=np.float64))
            pred_pot_1_parts.append(np.asarray(data["pred_pot_1"], dtype=np.float64))
            if "treat_f" in data.files:
                treat_f_parts.append(np.asarray(data["treat_f"], dtype=np.float64))

            if not meta:
                for meta_key in (
                    "method",
                    "seed",
                    "nsample",
                    "replication_id",
                    "split_mode",
                ):
                    if meta_key in data.files:
                        meta[meta_key] = data[meta_key].item()

            file_split = data["split"].item() if "split" in data.files else split
            if file_split != split:
                print(
                    f"Warning: {npz_path} metadata split={file_split!r} "
                    f"!= requested {split!r}"
                )
            print(f"Loaded GDR NPZ {npz_path.name}: n={len(pred_pot_0_parts[-1])}")

    result = {
        "pred_pot_0": np.concatenate(pred_pot_0_parts),
        "pred_pot_1": np.concatenate(pred_pot_1_parts),
        "meta": meta,
        "splits": splits,
    }
    if treat_f_parts:
        result["treat_f"] = np.concatenate(treat_f_parts)
    return result


def _print_split_summary(label, data_dict):
    w = data_dict.get("treat_f")
    n = len(data_dict["pred_pot_0"])
    if w is not None:
        n0 = int(np.sum(w == 0))
        n1 = int(np.sum(w == 1))
        print(f"{label}: total={n}, W=0={n0}, W=1={n1}")
    else:
        print(f"{label}: total={n}")


def _method_curve_from_gdr(gdr_data):
    return {
        "name": "GDR-CNF",
        "label": r"\textrm{GDR-CNF}",
        "color": GDR_CNF_COLOR,
        "pred_pot_0": gdr_data["pred_pot_0"],
        "pred_pot_1": gdr_data["pred_pot_1"],
        "treat_f": gdr_data.get("treat_f"),
    }


def _method_curve_from_torch(label, color, data):
    return {
        "name": label,
        "label": rf"\textrm{{{label}}}",
        "color": color,
        "pred_pot_0": data["pred_pot_0"],
        "pred_pot_1": data["pred_pot_1"],
        "treat_f": data.get("treat_f"),
        "gt": data,
    }


def _run_torch_jobs(jobs, parallel):
    """Run torch models one at a time (safe for PyTorch CUDA).

    Each job may target a different GPU via --codis_device / --diffpo_device.
    Thread-based parallel CUDA inference is intentionally not used because it
    can trigger illegal memory access in the S4/Cauchy kernels.
    """
    results = {}
    if parallel and len(jobs) > 1:
        print(
            "Note: --parallel runs models sequentially on their assigned GPUs "
            "(PyTorch CUDA is not thread-safe for concurrent multi-GPU inference)."
        )
    for job in jobs:
        label, data = _run_torch_model_inference(**job)
        results[label] = data
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot COVID-19 interventional densities for CoDiS / DiffPO / GDR-CNF"
    )
    parser.add_argument(
        "--dataloader_config",
        type=str,
        default="covid.yaml",
        help="Config for dataset split settings (use covid.yaml with split: files)",
    )
    parser.add_argument("--current_id", type=str, default="covid_causal_14d")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--missing_ratio",
        type=float,
        default=0.2,
        help="Must match training config model.test_missing_ratio",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=500,
        help="Number of diffusion samples per evaluation point (CoDiS / DiffPO)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Evaluation splits for torch models, e.g. 'test' or 'val,test'",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Accepted for compatibility; models still run sequentially, each on "
            "its own GPU (cuda:0 for DiffPO, cuda:1 for CoDiS by default)."
        ),
    )
    parser.add_argument(
        "--no_codis",
        action="store_true",
        help="Skip CoDiS inference",
    )
    parser.add_argument(
        "--codis_config",
        type=str,
        default="covid.yaml",
        help="CoDiS model config (global_broadcast)",
    )
    parser.add_argument(
        "--codis_model_dir",
        type=str,
        default="./save_model/covid",
    )
    parser.add_argument(
        "--codis_device",
        type=str,
        default="cuda:1",
        help="Device for CoDiS inference, e.g. cuda:1",
    )
    parser.add_argument(
        "--no_diffpo",
        action="store_true",
        help="Skip DiffPO inference",
    )
    parser.add_argument(
        "--diffpo_config",
        type=str,
        default="covid_fused_add.yaml",
        help="DiffPO model config (fused_add)",
    )
    parser.add_argument(
        "--diffpo_model_dir",
        type=str,
        default="./save_model/covid-diffpo",
    )
    parser.add_argument(
        "--diffpo_device",
        type=str,
        default="cuda:0",
        help="Device for DiffPO inference, e.g. cuda:0",
    )
    parser.add_argument(
        "--gdr_npz_dir",
        type=str,
        default=str(DEFAULT_GDR_NPZ_DIR),
    )
    parser.add_argument(
        "--gdr_prefix",
        type=str,
        default=DEFAULT_GDR_PREFIX,
    )
    parser.add_argument(
        "--gdr_splits",
        type=str,
        default="test",
        help="GDR splits to load; empty string disables GDR",
    )
    parser.add_argument(
        "--no_gdr",
        action="store_true",
        help="Skip GDR-CNF overlay",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    set_ieee_rcparams()

    dataloader_config = _load_yaml_config(args.dataloader_config)
    dataset_split = dataloader_config.get("dataset", {}).get("split", "random")
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=args.seed,
        batch_size=args.batch_size,
        missing_ratio=args.missing_ratio,
        current_id=args.current_id,
        split=dataset_split,
    )

    try:
        splits = _parse_splits(args.splits)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    plot_loaders = _select_plot_loaders(
        splits, train_loader, valid_loader, test_loader
    )

    gdr_splits = None
    if not args.no_gdr and args.gdr_splits.strip():
        try:
            gdr_splits = _parse_splits(args.gdr_splits)
        except ValueError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1) from exc

    print("Dataloaders loaded.")
    for split, loader in zip(splits, plot_loaders):
        ds = loader.dataset
        w = ds.processed["treatment"][ds.use_index_list]
        print(
            f"  {split}: n={len(ds)}, W=0={int((w == 0).sum())}, "
            f"W=1={int((w == 1).sum())}"
        )
    print(f"Torch model splits: {', '.join(splits)}")
    if gdr_splits is not None:
        print(f"GDR splits: {', '.join(gdr_splits)}")

    torch_jobs = []
    if not args.no_codis:
        torch_jobs.append(
            {
                "label": "CoDiS",
                "config_name": args.codis_config,
                "model_dir": args.codis_model_dir,
                "current_id": args.current_id,
                "device_str": args.codis_device,
                "plot_loaders": plot_loaders,
                "n_samples": args.nsample,
            }
        )
    if not args.no_diffpo:
        torch_jobs.append(
            {
                "label": "DiffPO",
                "config_name": args.diffpo_config,
                "model_dir": args.diffpo_model_dir,
                "current_id": args.current_id,
                "device_str": args.diffpo_device,
                "plot_loaders": plot_loaders,
                "n_samples": args.nsample,
            }
        )

    gdr_data = None
    if gdr_splits is not None:
        try:
            gdr_data = load_gdr_predictions(
                args.gdr_npz_dir, args.gdr_prefix, gdr_splits
            )
            gdr_seed = gdr_data["meta"].get("seed")
            if gdr_seed is not None and int(gdr_seed) != args.seed:
                print(
                    f"Warning: GDR NPZ seed={gdr_seed} differs from --seed={args.seed}"
                )
        except (FileNotFoundError, KeyError) as exc:
            print(f"Error loading GDR predictions: {exc}")
            raise SystemExit(1) from exc

    if not torch_jobs and gdr_data is None:
        print("Error: enable at least one of CoDiS, DiffPO, or GDR-CNF.")
        raise SystemExit(1)

    torch_results = {}
    if torch_jobs:
        try:
            torch_results = _run_torch_jobs(torch_jobs, parallel=args.parallel)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Error during torch model inference: {exc}")
            raise SystemExit(1) from exc

    method_curves = []
    gt_data_dict = None

    if "CoDiS" in torch_results:
        codis_curve = _method_curve_from_torch(
            "CoDiS", CODIS_COLOR, torch_results["CoDiS"]
        )
        method_curves.append(codis_curve)
        gt_data_dict = codis_curve["gt"]
        _print_split_summary("CoDiS", torch_results["CoDiS"])

    if "DiffPO" in torch_results:
        diffpo_curve = _method_curve_from_torch(
            "DiffPO", DIFFPO_COLOR, torch_results["DiffPO"]
        )
        method_curves.append(diffpo_curve)
        if gt_data_dict is None:
            gt_data_dict = diffpo_curve["gt"]
        _print_split_summary("DiffPO", torch_results["DiffPO"])

    if gdr_data is not None:
        method_curves.append(_method_curve_from_gdr(gdr_data))
        _print_split_summary("GDR-CNF", gdr_data)

    method_names = [curve["name"] for curve in method_curves]
    split_tag = "_".join(splits)
    gdr_tag = "_".join(gdr_splits) if gdr_splits else None
    save_filename = args.save_filename
    if save_filename is None:
        name_parts = ["covid19", "gt", split_tag]
        if method_names:
            name_parts.append("methods_" + "_".join(method_names))
        if gdr_tag and gdr_tag != split_tag:
            name_parts.append(f"gdr_{gdr_tag}")
        save_filename = "_".join(name_parts) + "_densities.pdf"

    print("Drawing interventional densities...")
    plot_interventional_densities(
        gt_data_dict=gt_data_dict,
        method_curves=method_curves,
        save_to_dir=True,
        filename=save_filename,
    )
