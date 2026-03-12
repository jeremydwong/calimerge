from __future__ import annotations

"""Visualization functions for extrinsic perturbation study results.

All functions are standalone pure functions. They read CSV files produced by the
perturbation sweep / Monte Carlo pipeline and write PNG images.
"""

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DPI = 150
_TITLE_SIZE = 13
_LABEL_SIZE = 11
_TICK_SIZE = 9

_ROT_PARAMS = ["rx", "ry", "rz"]
_TRANS_PARAMS = ["tx", "ty", "tz"]

# ---------------------------------------------------------------------------
# Sweep grid plot
# ---------------------------------------------------------------------------


def plot_sweep_grid(results_csv_path: Path, output_path: Path) -> None:
    """Create a 2x3 subplot grid showing sensitivity to each extrinsic parameter.

    Row 1: rotation parameters (rx, ry, rz)
    Row 2: translation parameters (tx, ty, tz)

    Each subplot plots mean and P95 3D error vs perturbation magnitude on
    log-log axes.

    Parameters
    ----------
    results_csv_path : Path
        Path to ``sweep_results.csv`` with columns:
        param, magnitude, unit, sign, mean_cm, median_cm, p95_cm, max_cm, ...
    output_path : Path
        Destination PNG file.
    """
    results_csv_path = Path(results_csv_path)
    output_path = Path(output_path)

    if not results_csv_path.exists():
        logger.warning("Sweep CSV not found: %s — skipping plot", results_csv_path)
        return

    df = pd.read_csv(results_csv_path)

    # Average the + and - sign results for each (param, magnitude) pair
    grouped = (
        df.groupby(["param", "magnitude", "unit"], as_index=False)
        .agg({"mean_cm": "mean", "p95_cm": "mean"})
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    all_params = _ROT_PARAMS + _TRANS_PARAMS
    row_labels = {
        0: "Rotation (degrees)",
        1: "Translation (mm)",
    }

    for idx, param in enumerate(all_params):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        subset = grouped[grouped["param"] == param].sort_values("magnitude")

        if subset.empty:
            ax.set_visible(False)
            continue

        magnitudes = subset["magnitude"].values
        mean_err = subset["mean_cm"].values
        p95_err = subset["p95_cm"].values

        # Filter out zero magnitudes for log-log (baseline)
        mask = magnitudes > 0
        if mask.sum() == 0:
            ax.text(
                0.5, 0.5, f"{param}\n(no non-zero data)",
                ha="center", va="center", transform=ax.transAxes, fontsize=_LABEL_SIZE,
            )
            continue

        mag = magnitudes[mask]
        mean_e = mean_err[mask]
        p95_e = p95_err[mask]

        ax.loglog(mag, mean_e, "o-", color="#2078B4", linewidth=1.5, markersize=4, label="Mean")
        ax.loglog(mag, p95_e, "s--", color="#D62728", linewidth=1.5, markersize=4, label="P95")

        ax.set_title(param, fontsize=_TITLE_SIZE, fontweight="bold")
        ax.set_ylabel("3D Error (cm)", fontsize=_LABEL_SIZE)
        ax.set_xlabel(row_labels[row], fontsize=_LABEL_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=_TICK_SIZE, loc="upper left")

    fig.suptitle("Extrinsic Perturbation Sensitivity", fontsize=_TITLE_SIZE + 2, fontweight="bold")

    fig.savefig(str(output_path), dpi=_DPI)
    plt.close(fig)
    logger.info("Saved sweep grid plot to %s", output_path)


# ---------------------------------------------------------------------------
# Monte Carlo box plots
# ---------------------------------------------------------------------------


def plot_monte_carlo_boxes(results_csv_path: Path, output_path: Path) -> None:
    """Create box plots of Monte Carlo perturbation trial errors grouped by sigma.

    Parameters
    ----------
    results_csv_path : Path
        Path to ``monte_carlo_results.csv`` with columns:
        trial, sigma_rot_deg, sigma_trans_mm, mean_cm, median_cm, p95_cm, max_cm, ...
    output_path : Path
        Destination PNG file.
    """
    results_csv_path = Path(results_csv_path)
    output_path = Path(output_path)

    if not results_csv_path.exists():
        logger.warning("Monte Carlo CSV not found: %s — skipping plot", results_csv_path)
        return

    df = pd.read_csv(results_csv_path)

    # Build a readable label for each sigma level
    df["sigma_label"] = (
        df["sigma_rot_deg"].apply(lambda v: f"{v:g}")
        + "\u00b0 / "
        + df["sigma_trans_mm"].apply(lambda v: f"{v:g}")
        + "mm"
    )

    # Sort sigma levels by rotation magnitude so box plots are ordered
    label_order = (
        df[["sigma_rot_deg", "sigma_trans_mm", "sigma_label"]]
        .drop_duplicates()
        .sort_values("sigma_rot_deg")
    )
    ordered_labels = label_order["sigma_label"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Prepare data grouped by sigma label in order
    box_data = []
    positions = []
    tick_labels = []
    for i, label in enumerate(ordered_labels):
        subset = df[df["sigma_label"] == label]["mean_cm"].values
        box_data.append(subset)
        positions.append(i + 1)
        tick_labels.append(label)

    if not box_data:
        plt.close(fig)
        logger.warning("No Monte Carlo data to plot")
        return

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )

    # Style the boxes
    box_color = "#AEC7E8"
    median_color = "#D62728"
    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_edgecolor("#2078B4")
    for median_line in bp["medians"]:
        median_line.set_color(median_color)
        median_line.set_linewidth(2)

    # Overlay individual trial points with jitter
    rng = np.random.default_rng(42)
    for i, (pos, data) in enumerate(zip(positions, box_data)):
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(
            pos + jitter,
            data,
            alpha=0.5,
            s=18,
            color="#FF7F0E",
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=_LABEL_SIZE)
    ax.set_ylabel("3D Error (cm)", fontsize=_LABEL_SIZE)
    ax.set_xlabel("Sigma Level (rotation / translation)", fontsize=_LABEL_SIZE)
    ax.set_title(
        "Monte Carlo Perturbation: Error Distribution",
        fontsize=_TITLE_SIZE,
        fontweight="bold",
    )
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(str(output_path), dpi=_DPI)
    plt.close(fig)
    logger.info("Saved Monte Carlo box plot to %s", output_path)


# ---------------------------------------------------------------------------
# Per-keypoint sensitivity heatmap
# ---------------------------------------------------------------------------


def plot_keypoint_heatmap(sensitivity_csv_path: Path, output_path: Path) -> None:
    """Create a heatmap of per-keypoint sensitivity to each perturbation parameter.

    Parameters
    ----------
    sensitivity_csv_path : Path
        Path to ``per_keypoint_sensitivity.csv`` with columns:
        keypoint, rx_1deg_cm, ry_1deg_cm, rz_1deg_cm, tx_10mm_cm, ty_10mm_cm, tz_10mm_cm
    output_path : Path
        Destination PNG file.
    """
    sensitivity_csv_path = Path(sensitivity_csv_path)
    output_path = Path(output_path)

    if not sensitivity_csv_path.exists():
        logger.warning(
            "Per-keypoint sensitivity CSV not found: %s — skipping plot",
            sensitivity_csv_path,
        )
        return

    df = pd.read_csv(sensitivity_csv_path)

    keypoint_col = df.columns[0]  # "keypoint"
    value_cols = [c for c in df.columns if c != keypoint_col]

    keypoints = df[keypoint_col].tolist()
    data = df[value_cols].values.astype(float)

    # Clean up column names for display (e.g. "rx_1deg_cm" -> "rx 1deg")
    col_labels = []
    for c in value_cols:
        label = c.replace("_cm", "").replace("_", " ")
        col_labels.append(label)

    fig, ax = plt.subplots(
        figsize=(max(8, len(value_cols) * 1.4), max(6, len(keypoints) * 0.45)),
        constrained_layout=True,
    )

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Add text annotations in each cell
    for i in range(len(keypoints)):
        for j in range(len(value_cols)):
            val = data[i, j]
            # Use dark text on light cells and light text on dark cells
            text_color = "white" if val > (data.max() * 0.65) else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=_TICK_SIZE,
                color=text_color,
            )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=_LABEL_SIZE, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(keypoints)))
    ax.set_yticklabels(keypoints, fontsize=_TICK_SIZE)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Error (cm)", fontsize=_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=_TICK_SIZE)

    ax.set_title(
        "Per-Keypoint Sensitivity to Extrinsic Perturbation",
        fontsize=_TITLE_SIZE,
        fontweight="bold",
    )

    fig.savefig(str(output_path), dpi=_DPI)
    plt.close(fig)
    logger.info("Saved keypoint heatmap to %s", output_path)


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------


def generate_all_plots(output_dir: Path) -> None:
    """Generate all perturbation study plots from CSVs in *output_dir*.

    Looks for:
      - ``sweep_results.csv``         -> ``sweep_sensitivity.png``
      - ``monte_carlo_results.csv``   -> ``monte_carlo_boxes.png``
      - ``per_keypoint_sensitivity.csv`` -> ``keypoint_heatmap.png``

    Missing CSVs are silently skipped.
    """
    output_dir = Path(output_dir)

    sweep_csv = output_dir / "sweep_results.csv"
    mc_csv = output_dir / "monte_carlo_results.csv"
    kp_csv = output_dir / "per_keypoint_sensitivity.csv"

    if sweep_csv.exists():
        plot_sweep_grid(sweep_csv, output_dir / "sweep_sensitivity.png")
    if mc_csv.exists():
        plot_monte_carlo_boxes(mc_csv, output_dir / "monte_carlo_boxes.png")
    if kp_csv.exists():
        plot_keypoint_heatmap(kp_csv, output_dir / "keypoint_heatmap.png")
