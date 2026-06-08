from __future__ import annotations

import os
import re
from itertools import cycle
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sanitize_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("+", "plus")
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _safe_divide(numerator, denominator):
    """
    Elementwise division that returns NaN where denominator == 0.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    out = np.full_like(numerator, np.nan, dtype=float)
    mask = denominator != 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _load_x_y_from_csv(
    csv_path: str,
    *,
    y_col: str = "f1",
    frames_per_ns: Optional[float] = None,
    time_scale: float = 10.0,
):
    """
    Load one metrics CSV and return x, y, x_label, and dataframe.

    Supported y_col values include:
      - "f1"
      - "bce"
      - "incorrect_predictions"                  -> fp + fn
      - "incorrect_predictions_per_npixels"     -> (fp + fn) / n_pixels
      - "incorrect_predictions_over_predunion"  -> (fp + fn) / (tp + fp + fn)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    derived_requirements = {
        "incorrect_predictions": {"frame_id", "fp", "fn"},
        "incorrect_predictions_per_npixels": {"frame_id", "fp", "fn", "n_pixels"},
        "incorrect_predictions_over_predunion": {"frame_id", "tp", "fp", "fn"},
    }

    if y_col in derived_requirements:
        required_cols = derived_requirements[y_col]
    else:
        required_cols = {"frame_id", y_col}

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {csv_path}")

    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")

    if y_col == "incorrect_predictions":
        df["fp"] = pd.to_numeric(df["fp"], errors="coerce")
        df["fn"] = pd.to_numeric(df["fn"], errors="coerce")
        df["incorrect_predictions"] = df["fp"] + df["fn"]

    elif y_col == "incorrect_predictions_per_npixels":
        df["fp"] = pd.to_numeric(df["fp"], errors="coerce")
        df["fn"] = pd.to_numeric(df["fn"], errors="coerce")
        df["n_pixels"] = pd.to_numeric(df["n_pixels"], errors="coerce")
        incorrect = df["fp"] + df["fn"]
        df["incorrect_predictions_per_npixels"] = _safe_divide(
            incorrect.to_numpy(),
            df["n_pixels"].to_numpy(),
        )

    elif y_col == "incorrect_predictions_over_predunion":
        df["tp"] = pd.to_numeric(df["tp"], errors="coerce")
        df["fp"] = pd.to_numeric(df["fp"], errors="coerce")
        df["fn"] = pd.to_numeric(df["fn"], errors="coerce")
        incorrect = df["fp"] + df["fn"]
        denom = df["tp"] + df["fp"] + df["fn"]
        df["incorrect_predictions_over_predunion"] = _safe_divide(
            incorrect.to_numpy(),
            denom.to_numpy(),
        )

    else:
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    df = df.dropna(subset=["frame_id", y_col]).sort_values("frame_id").reset_index(drop=True)

    if "time_ns" in df.columns:
        df["time_ns"] = pd.to_numeric(df["time_ns"], errors="coerce")
        if df["time_ns"].notna().any():
            x = df["time_ns"].to_numpy() * time_scale
        else:
            if frames_per_ns is None:
                x = df["frame_id"].to_numpy() * time_scale
            else:
                x = (df["frame_id"].to_numpy() / float(frames_per_ns)) * time_scale
    else:
        if frames_per_ns is None:
            x = df["frame_id"].to_numpy() * time_scale
        else:
            x = (df["frame_id"].to_numpy() / float(frames_per_ns)) * time_scale

    y = df[y_col].to_numpy()
    x_label = "Time (ns)"
    return x, y, x_label, df


def _extract_metric_at_time(
    csv_path: str,
    *,
    target_time_ns: float,
    y_col: str = "f1",
):
    """
    Load one metrics CSV and return the value of y_col at the row whose time_ns
    is closest to target_time_ns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    required = {"time_ns", y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {csv_path}")

    df["time_ns"] = pd.to_numeric(df["time_ns"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["time_ns", y_col]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No valid rows with time_ns and {y_col} in: {csv_path}")

    idx = (df["time_ns"] - float(target_time_ns)).abs().idxmin()
    row = df.loc[idx]

    return {
        "value": float(row[y_col]),
        "time_ns_used": float(row["time_ns"]),
        "row_index": int(idx),
    }


def _extract_confusion_counts_from_csv(
    csv_path: str,
    *,
    use_totals: bool = True,
    target_time_ns: Optional[float] = None,
    count_cols: Sequence[str] = ("tp", "fp", "fn"),
):
    """
    Extract TP/FP/FN counts from one per_frame_metrics.csv.

    If use_totals=True:
        sums TP/FP/FN over all frames.

    If use_totals=False:
        uses the row closest to target_time_ns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    required = set(count_cols)
    if not use_totals:
        required = required | {"time_ns"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {csv_path}")

    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if use_totals:
        counts = {c: float(df[c].sum()) for c in count_cols}
        time_ns_used = None
        row_index = None
    else:
        if target_time_ns is None:
            raise ValueError("target_time_ns must be provided when use_totals=False")

        df["time_ns"] = pd.to_numeric(df["time_ns"], errors="coerce")
        df = df.dropna(subset=["time_ns"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No valid rows with time_ns in: {csv_path}")

        idx = (df["time_ns"] - float(target_time_ns)).abs().idxmin()
        row = df.loc[idx]

        counts = {c: float(row[c]) for c in count_cols}
        time_ns_used = float(row["time_ns"])
        row_index = int(idx)

    return {
        "counts": counts,
        "time_ns_used": time_ns_used,
        "row_index": row_index,
    }



def _extract_classification_metrics_from_csv(
    csv_path: str,
    *,
    use_totals: bool = True,
    target_time_ns: Optional[float] = None,
):
    """
    Extract F1, recall, and precision from one per_frame_metrics.csv.

    If use_totals=True:
        sums TP/FP/FN over all frames, then computes global full-simulation metrics.

    If use_totals=False:
        uses the row closest to target_time_ns and computes metrics from that row.
    """
    info = _extract_confusion_counts_from_csv(
        csv_path,
        use_totals=use_totals,
        target_time_ns=target_time_ns,
        count_cols=("tp", "fp", "fn"),
    )

    counts = info["counts"]

    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else np.nan

    return {
        "metrics": {
            "f1": f1,
            "recall": recall,
            "precision": precision,
        },
        "counts": counts,
        "time_ns_used": info["time_ns_used"],
        "row_index": info["row_index"],
    }

def _plot_subset(
    csv_map_subset: Dict[str, Dict[str, str]],
    save_path: str,
    *,
    title: str,
    y_col: str = "f1",
    y_label: Optional[str] = None,
    y_lim: Optional[tuple] = None,
    frames_per_ns: Optional[float] = None,
    time_scale: float = 10.0,
    show: bool = False,
    linewidth: float = 2.0,
    colors: Optional[Sequence[str]] = None,
):
    """
    Internal helper to plot one subset of curves and save the figure.
    """
    loaded = []

    plt.figure(figsize=(14, 8))
    x_label = "Frame ID"

    if colors is not None and len(colors) > 0:
        color_cycle = cycle(colors)
    else:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        color_cycle = cycle(default_colors if default_colors else [None])

    for model_name, velocity_dict in csv_map_subset.items():
        for velocity_label, csv_path in velocity_dict.items():
            x, y, x_label, df = _load_x_y_from_csv(
                csv_path,
                y_col=y_col,
                frames_per_ns=frames_per_ns,
                time_scale=time_scale,
            )

            color = next(color_cycle)
            label = f"{model_name}, {velocity_label}"

            plt.plot(
                x,
                y,
                label=label,
                linestyle="-",
                color=color,
                linewidth=linewidth,
            )

            loaded.append(
                {
                    "model": model_name,
                    "velocity": velocity_label,
                    "csv_path": csv_path,
                    "n_points": len(df),
                    "x_max": float(x.max()) if len(x) else None,
                    "color": color,
                    "y_col": y_col,
                }
            )

    plt.xlabel(x_label, fontsize=28)
    plt.ylabel(y_label if y_label is not None else y_col, fontsize=28)
    plt.title(title, fontsize=24)

    if y_lim is not None:
        plt.ylim(*y_lim)

    plt.xlim(0.0, 30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved plot to: {save_path}")
    return loaded


def _plot_individual_simulations(
    csv_map: Dict[str, Dict[str, str]],
    out_dir: str,
    *,
    y_col: str = "f1",
    y_label: Optional[str] = None,
    y_lim: Optional[tuple] = None,
    start_from_positive_only: bool = False,
    frames_per_ns: Optional[float] = None,
    time_scale: float = 10.0,
    show: bool = False,
    linewidth: float = 2.0,
    color: Optional[str] = None,
):
    """
    Make one metric-vs-time plot per simulation.
    Saves 1 plot per CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for model_name, velocity_dict in csv_map.items():
        for velocity_label, csv_path in velocity_dict.items():
            x, y, x_label, df = _load_x_y_from_csv(
                csv_path,
                y_col=y_col,
                frames_per_ns=frames_per_ns,
                time_scale=time_scale,
            )

            positive_mask = y > 0

            if start_from_positive_only:
                if positive_mask.any():
                    first_idx = positive_mask.argmax()
                    x_plot = x[first_idx:]
                    y_plot = y[first_idx:]
                else:
                    x_plot = []
                    y_plot = []
            else:
                x_plot = x
                y_plot = y

            model_slug = _sanitize_name(model_name)
            velocity_slug = _sanitize_name(velocity_label)
            save_name = f"{_sanitize_name(y_col)}_over_time_{model_slug}_{velocity_slug}.png"
            save_path = os.path.join(out_dir, save_name)

            plt.figure(figsize=(11, 6))
            if len(x_plot) > 0:
                plt.plot(
                    x_plot,
                    y_plot,
                    linestyle="-",
                    color=color,
                    linewidth=linewidth,
                )

            plt.xlabel(x_label, fontsize=18)
            plt.ylabel(y_label if y_label is not None else y_col, fontsize=18)

            if y_lim is not None:
                plt.ylim(*y_lim)

            plt.xlim(0, 30)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()

            print(f"Saved individual plot to: {save_path}")
            saved_paths.append(
                {
                    "model": model_name,
                    "velocity": velocity_label,
                    "csv_path": csv_path,
                    "plot_path": save_path,
                    "started_from_positive_only": start_from_positive_only,
                    "had_positive_values": bool(positive_mask.any()),
                    "color": color,
                    "y_col": y_col,
                }
            )

    return saved_paths


def plot_training_loss_comparison(
    loss_csv_map: Dict[str, str],
    out_dir: str,
    *,
    out_name_loss: str = "training_and_validation_loss_comparison.png",
    out_name_val_loss: str = "training_and_validation_loss_comparison.png",
    title_loss: str = "Training and Validation Loss Comparison",
    title_val_loss: str = "Training and Validation Loss Comparison",
    max_epoch: Optional[float] = None,
    show: bool = False,
    linewidth: float = 2.0,
    colors: Optional[Sequence[str]] = None,
):
    """
    Plot training and validation loss curves on the same figure.

    Each CSV should contain:
      - epoch
      - loss
    and may also contain:
      - val_loss

    Line styles:
      - solid  : training loss
      - dashed : validation loss
    """
    os.makedirs(out_dir, exist_ok=True)

    save_loss = os.path.join(out_dir, out_name_loss)

    if colors is not None and len(colors) > 0:
        plot_colors = list(colors)
    else:
        plot_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not plot_colors:
            plot_colors = [None]

    loaded = []
    has_any_val_loss = False

    plt.figure(figsize=(11, 6))
    color_cycle = cycle(plot_colors)

    for model_name, csv_path in loss_csv_map.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find loss CSV: {csv_path}")

        df = pd.read_csv(csv_path).copy()

        if "epoch" not in df.columns or "loss" not in df.columns:
            raise ValueError(
                f"Loss CSV must contain 'epoch' and 'loss' columns: {csv_path}"
            )

        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")

        has_val_loss_this_model = "val_loss" in df.columns
        if has_val_loss_this_model:
            df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")

        df = df.sort_values("epoch").reset_index(drop=True)
        df_train = df.dropna(subset=["epoch", "loss"]).copy()

        if max_epoch is not None:
            df_train = df_train[df_train["epoch"] <= max_epoch].copy()

        color = next(color_cycle)

        plt.plot(
            df_train["epoch"].to_numpy(),
            df_train["loss"].to_numpy(),
            label=f"{model_name} train",
            linestyle="-",
            color=color,
            linewidth=linewidth,
        )

        val_points = 0
        if has_val_loss_this_model:
            df_val = df.dropna(subset=["epoch", "val_loss"]).copy()

            if max_epoch is not None:
                df_val = df_val[df_val["epoch"] <= max_epoch].copy()

            if len(df_val) > 0:
                has_any_val_loss = True
                val_points = len(df_val)

                plt.plot(
                    df_val["epoch"].to_numpy(),
                    df_val["val_loss"].to_numpy(),
                    label=f"{model_name} val",
                    linestyle="--",
                    color=color,
                    linewidth=linewidth,
                )

        loaded.append(
            {
                "model": model_name,
                "csv_path": csv_path,
                "n_train_points": len(df_train),
                "n_val_points": val_points,
                "has_val_loss": has_val_loss_this_model,
                "color": color,
                "max_epoch": max_epoch,
            }
        )

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title(title_loss, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if max_epoch is not None:
        plt.xlim(0, max_epoch)

    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize = 18)
    plt.tight_layout()
    plt.savefig(save_loss, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved combined training/validation loss plot to: {save_loss}")

    return {
        "training_loss_plot_path": save_loss,
        "validation_loss_plot_path": save_loss,
        "loaded_loss_series": loaded,
        "has_any_val_loss": has_any_val_loss,
    }


def plot_metric_bar_at_time(
    csv_map: Dict[str, Dict[str, str]],
    out_dir: str,
    *,
    target_time_ns: float,
    y_col: str = "f1",
    out_name: str = "f1_bar_comparison_at_time.png",
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = False,
    model_order: Optional[Sequence[str]] = None,
    bar_colors: Optional[Sequence[str]] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
):
    """
    Create a grouped bar chart at a requested time.
    X-axis groups are the inner csv_map keys.
    Bars within each group are models.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name)

    if model_order is None:
        model_order = [k for k in ["Base", "Base + SED"] if k in csv_map]
        for k in csv_map:
            if k not in model_order:
                model_order.append(k)

    if len(model_order) == 0:
        raise ValueError("No models found in csv_map.")

    velocity_labels = []
    for model_name in model_order:
        for vel in csv_map.get(model_name, {}).keys():
            if vel not in velocity_labels:
                velocity_labels.append(vel)

    if len(velocity_labels) == 0:
        raise ValueError("No entries found in csv_map.")

    display_velocity_labels = [str(v).replace("m/s", "").strip() for v in velocity_labels]

    n_models = len(model_order)
    x = np.arange(len(velocity_labels))
    width = 0.8 / max(n_models, 1)

    if bar_colors is None or len(bar_colors) == 0:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_colors) >= n_models:
            bar_colors = default_colors[:n_models]
        else:
            bar_colors = [None] * n_models

    plt.figure(figsize=figsize)

    loaded = []

    for i, model_name in enumerate(model_order):
        values = []

        for vel in velocity_labels:
            csv_path = csv_map.get(model_name, {}).get(vel, None)

            if csv_path is None:
                values.append(np.nan)
                continue

            info = _extract_metric_at_time(
                csv_path,
                target_time_ns=target_time_ns,
                y_col=y_col,
            )
            values.append(info["value"])

            print(
                f"{model_name} | {vel} | {y_col.upper()} at requested {target_time_ns:.2f} ns "
                f"(using {info['time_ns_used']:.2f} ns) = {info['value']:.4f}"
            )

            loaded.append(
                {
                    "model": model_name,
                    "velocity": vel,
                    "csv_path": csv_path,
                    "target_time_ns": float(target_time_ns),
                    "time_ns_used": info["time_ns_used"],
                    "value": info["value"],
                    "y_col": y_col,
                }
            )

        offsets = x + (i - (n_models - 1) / 2.0) * width
        color = bar_colors[i] if i < len(bar_colors) else None

        plt.bar(
            offsets,
            values,
            width=width,
            label=model_name,
            color=color,
            zorder=3,
        )

        plt.grid(True, axis="y", alpha=0.3, zorder=0)
        plt.gca().set_axisbelow(True)

    plt.xlabel("Velocity / test case", fontsize=18)
    plt.ylabel(y_label if y_label is not None else y_col, fontsize=18)

    if title is None:
        plt.title(f"{y_col.upper()} at {target_time_ns:.2f} ns", fontsize=18)
    else:
        plt.title(title, fontsize=18)

    plt.xticks(x, display_velocity_labels, fontsize=16)
    plt.yticks(fontsize=16)

    if y_col.lower() == "f1":
        plt.ylim(0.0, 1.05)

    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved bar chart to: {save_path}")

    return {
        "bar_plot_path": save_path,
        "target_time_ns": float(target_time_ns),
        "loaded_bar_values": loaded,
        "velocity_labels": velocity_labels,
        "display_velocity_labels": display_velocity_labels,
        "model_order": list(model_order),
    }


def plot_confusion_counts_bar_by_geometry(
    csv_map: Dict[str, Dict[str, str]],
    out_dir: str,
    *,
    out_name: str = "tp_fp_fn_bar_by_geometry.png",
    title: Optional[str] = None,
    show: bool = False,
    model_order: Optional[Sequence[str]] = None,
    geometry_order: Optional[Sequence[str]] = None,
    use_totals: bool = True,
    target_time_ns: Optional[float] = None,
    count_cols: Sequence[str] = ("tp", "fp", "fn"),
    count_labels: Optional[Dict[str, str]] = None,
    colors: Optional[Sequence[str]] = None,
    xlabel: str = "Test geometry",
    ylabel: str = "Count",
    figsize: tuple = (12, 7),
    dpi: int = 300,
    rotate_xticks: float = 0,
    add_value_labels: bool = True,
):
    """
    Create one grouped bar plot with TP, FP, and FN for each geometry.

    Expected csv_map format for one model across different geometries:

    csv_map = {
        "Base": {
            "Similar": "/path/to/per_frame_metrics.csv",
            "Horizontal": "/path/to/per_frame_metrics.csv",
            "Vertical": "/path/to/per_frame_metrics.csv",
        }
    }

    X-axis groups are geometries.
    Within each geometry group, bars are TP, FP, FN.

    If multiple outer models are passed, x labels become:
        "Base: Similar", "Base + SED: Similar", etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name)

    if count_labels is None:
        count_labels = {
            "tp": "True Positives",
            "fp": "False Positives",
            "fn": "False Negatives",
        }

    if model_order is None:
        model_order = list(csv_map.keys())

    if len(model_order) == 0:
        raise ValueError("No models found in csv_map.")

    x_categories = []

    if geometry_order is not None:
        for model_name in model_order:
            if model_name not in csv_map:
                continue

            for geom in geometry_order:
                if geom not in csv_map[model_name]:
                    continue

                label = str(geom) if len(model_order) == 1 else f"{model_name}: {geom}"

                x_categories.append(
                    {
                        "model": model_name,
                        "geometry": geom,
                        "label": label,
                        "csv_path": csv_map[model_name][geom],
                    }
                )
    else:
        for model_name in model_order:
            if model_name not in csv_map:
                continue

            for geom, csv_path in csv_map[model_name].items():
                label = str(geom) if len(model_order) == 1 else f"{model_name}: {geom}"

                x_categories.append(
                    {
                        "model": model_name,
                        "geometry": geom,
                        "label": label,
                        "csv_path": csv_path,
                    }
                )

    if len(x_categories) == 0:
        raise ValueError("No geometry CSV entries found in csv_map.")

    loaded = []
    values_by_count_col = {c: [] for c in count_cols}

    for item in x_categories:
        info = _extract_confusion_counts_from_csv(
            item["csv_path"],
            use_totals=use_totals,
            target_time_ns=target_time_ns,
            count_cols=count_cols,
        )

        counts = info["counts"]

        for c in count_cols:
            values_by_count_col[c].append(counts[c])

        loaded_item = {
            "model": item["model"],
            "geometry": item["geometry"],
            "label": item["label"],
            "csv_path": item["csv_path"],
            "use_totals": use_totals,
            "target_time_ns": target_time_ns,
            "time_ns_used": info["time_ns_used"],
            "row_index": info["row_index"],
        }

        for c in count_cols:
            loaded_item[c] = counts[c]

        loaded.append(loaded_item)

        if use_totals:
            print(
                f"{item['label']} | "
                + " | ".join([f"{c.upper()} total = {counts[c]:,.0f}" for c in count_cols])
            )
        else:
            print(
                f"{item['label']} | requested {target_time_ns:.2f} ns "
                f"(using {info['time_ns_used']:.2f} ns) | "
                + " | ".join([f"{c.upper()} = {counts[c]:,.0f}" for c in count_cols])
            )

    x = np.arange(len(x_categories))
    n_bars = len(count_cols)
    width = 0.8 / max(n_bars, 1)

    if colors is None or len(colors) == 0:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_colors) >= n_bars:
            colors = default_colors[:n_bars]
        else:
            colors = [None] * n_bars

    fig, ax = plt.subplots(figsize=figsize)

    for i, c in enumerate(count_cols):
        offsets = x + (i - (n_bars - 1) / 2.0) * width
        color = colors[i] if i < len(colors) else None

        bars = ax.bar(
            offsets,
            values_by_count_col[c],
            width=width,
            label=count_labels.get(c, c.upper()),
            color=color,
            zorder=3,
        )

        if add_value_labels:
            for bar, value in zip(bars, values_by_count_col[c]):
                if np.isnan(value):
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    if title is None:
        if use_totals:
            title = "TP, FP, and FN Counts by Test Geometry"
        else:
            title = f"TP, FP, and FN Counts by Test Geometry at {target_time_ns:.2f} ns"

    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [item["label"] for item in x_categories],
        fontsize=18,
        rotation=rotate_xticks,
        ha="right" if rotate_xticks else "center",
    )
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=18)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved TP/FP/FN geometry bar plot to: {save_path}")

    return {
        "confusion_bar_plot_path": save_path,
        "loaded_confusion_counts": loaded,
        "x_labels": [item["label"] for item in x_categories],
        "count_cols": tuple(count_cols),
        "use_totals": use_totals,
        "target_time_ns": target_time_ns,
    }



def plot_classification_metrics_bar_by_geometry(
    csv_map: Dict[str, Dict[str, str]],
    out_dir: str,
    *,
    out_name: str = "f1_recall_precision_bar_by_geometry.png",
    title: Optional[str] = None,
    show: bool = False,
    model_order: Optional[Sequence[str]] = None,
    geometry_order: Optional[Sequence[str]] = None,
    use_totals: bool = True,
    target_time_ns: Optional[float] = None,
    metric_cols: Sequence[str] = ("f1", "recall", "precision"),
    metric_labels: Optional[Dict[str, str]] = None,
    colors: Optional[Sequence[str]] = None,
    xlabel: str = "Test geometry",
    ylabel: str = "Metric value",
    figsize: tuple = (12, 7),
    dpi: int = 300,
    rotate_xticks: float = 0,
    add_value_labels: bool = True,
):
    """
    Create one grouped bar plot with F1, recall, and precision for each geometry.

    Expected csv_map format for one model across different geometries:

    csv_map = {
        "ML": {
            "Similar": "/path/to/per_frame_metrics.csv",
            "Finer": "/path/to/per_frame_metrics.csv",
            "Vertical": "/path/to/per_frame_metrics.csv",
        }
    }

    X-axis groups are geometries. Within each geometry group, bars are F1,
    recall, and precision.

    If multiple outer models are passed, x labels become:
        "ML: Similar", "ML + SED: Similar", etc.

    Metrics are computed from summed TP/FP/FN over the full simulation when
    use_totals=True. This is usually better than frame-wise averaging because
    it gives one global full-rollout score per test case.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name)

    allowed_metrics = {"f1", "recall", "precision"}
    bad_metrics = set(metric_cols) - allowed_metrics
    if bad_metrics:
        raise ValueError(f"Unsupported metric_cols {bad_metrics}. Use only {allowed_metrics}.")

    if metric_labels is None:
        metric_labels = {
            "f1": "F1",
            "recall": "Recall",
            "precision": "Precision",
        }

    if model_order is None:
        model_order = list(csv_map.keys())

    if len(model_order) == 0:
        raise ValueError("No models found in csv_map.")

    x_categories = []

    if geometry_order is not None:
        for model_name in model_order:
            if model_name not in csv_map:
                continue

            for geom in geometry_order:
                if geom not in csv_map[model_name]:
                    continue

                label = str(geom) if len(model_order) == 1 else f"{model_name}: {geom}"

                x_categories.append(
                    {
                        "model": model_name,
                        "geometry": geom,
                        "label": label,
                        "csv_path": csv_map[model_name][geom],
                    }
                )
    else:
        for model_name in model_order:
            if model_name not in csv_map:
                continue

            for geom, csv_path in csv_map[model_name].items():
                label = str(geom) if len(model_order) == 1 else f"{model_name}: {geom}"

                x_categories.append(
                    {
                        "model": model_name,
                        "geometry": geom,
                        "label": label,
                        "csv_path": csv_path,
                    }
                )

    if len(x_categories) == 0:
        raise ValueError("No geometry CSV entries found in csv_map.")

    loaded = []
    values_by_metric = {m: [] for m in metric_cols}

    for item in x_categories:
        info = _extract_classification_metrics_from_csv(
            item["csv_path"],
            use_totals=use_totals,
            target_time_ns=target_time_ns,
        )

        metrics = info["metrics"]
        counts = info["counts"]

        for m in metric_cols:
            values_by_metric[m].append(metrics[m])

        loaded_item = {
            "model": item["model"],
            "geometry": item["geometry"],
            "label": item["label"],
            "csv_path": item["csv_path"],
            "use_totals": use_totals,
            "target_time_ns": target_time_ns,
            "time_ns_used": info["time_ns_used"],
            "row_index": info["row_index"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }

        for m in metric_cols:
            loaded_item[m] = metrics[m]

        loaded.append(loaded_item)

        if use_totals:
            print(
                f"{item['label']} | "
                f"F1 = {metrics['f1']:.4f} | "
                f"Recall = {metrics['recall']:.4f} | "
                f"Precision = {metrics['precision']:.4f} | "
                f"TP = {counts['tp']:,.0f} | FP = {counts['fp']:,.0f} | FN = {counts['fn']:,.0f}"
            )
        else:
            print(
                f"{item['label']} | requested {target_time_ns:.2f} ns "
                f"(using {info['time_ns_used']:.2f} ns) | "
                f"F1 = {metrics['f1']:.4f} | "
                f"Recall = {metrics['recall']:.4f} | "
                f"Precision = {metrics['precision']:.4f}"
            )

    x = np.arange(len(x_categories))
    n_bars = len(metric_cols)
    width = 0.8 / max(n_bars, 1)

    if colors is None or len(colors) == 0:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_colors) >= n_bars:
            colors = default_colors[:n_bars]
        else:
            colors = [None] * n_bars

    fig, ax = plt.subplots(figsize=figsize)

    for i, m in enumerate(metric_cols):
        offsets = x + (i - (n_bars - 1) / 2.0) * width
        color = colors[i] if i < len(colors) else None

        bars = ax.bar(
            offsets,
            values_by_metric[m],
            width=width,
            label=metric_labels.get(m, m.upper()),
            color=color,
            zorder=3,
        )

        if add_value_labels:
            for bar, value in zip(bars, values_by_metric[m]):
                if np.isnan(value):
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    rotation=0,
                )

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_ylim(0.0, 1.05)

    if title is None:
        if use_totals:
            title = "F1, Recall, and Precision by Test Geometry"
        else:
            title = f"F1, Recall, and Precision by Test Geometry at {target_time_ns:.2f} ns"

    #ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [item["label"] for item in x_categories],
        fontsize=20,
        rotation=rotate_xticks,
        ha="right" if rotate_xticks else "center",
    )
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=16)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved F1/Recall/Precision geometry bar plot to: {save_path}")

    return {
        "classification_metrics_bar_plot_path": save_path,
        "loaded_classification_metrics": loaded,
        "x_labels": [item["label"] for item in x_categories],
        "metric_cols": tuple(metric_cols),
        "use_totals": use_totals,
        "target_time_ns": target_time_ns,
    }

def plot_f1_full_simulation_comparison(
    csv_map: Dict[str, Dict[str, str]],
    out_dir: str,
    *,
    loss_csv_map: Optional[Dict[str, str]] = None,
    out_name_all: str = "f1_full_simulation_comparison.png",
    out_name_base: str = "f1_full_simulation_base_only.png",
    out_name_sed: str = "f1_full_simulation_base_plus_sed_only.png",
    out_name_bce_all: str = "bce_full_simulation_comparison.png",
    out_name_bce_base: str = "bce_full_simulation_base_only.png",
    out_name_bce_sed: str = "bce_full_simulation_base_plus_sed_only.png",
    out_name_incorrect_all: str = "incorrect_predictions_full_simulation_comparison.png",
    out_name_incorrect_base: str = "incorrect_predictions_full_simulation_base_only.png",
    out_name_incorrect_sed: str = "incorrect_predictions_full_simulation_base_plus_sed_only.png",
    out_name_incorrect_norm_npixels_all: str = "incorrect_predictions_per_npixels_full_simulation_comparison.png",
    out_name_incorrect_norm_npixels_base: str = "incorrect_predictions_per_npixels_full_simulation_base_only.png",
    out_name_incorrect_norm_npixels_sed: str = "incorrect_predictions_per_npixels_full_simulation_base_plus_sed_only.png",
    out_name_incorrect_over_predunion_all: str = "incorrect_predictions_over_predunion_full_simulation_comparison.png",
    out_name_incorrect_over_predunion_base: str = "incorrect_predictions_over_predunion_full_simulation_base_only.png",
    out_name_incorrect_over_predunion_sed: str = "incorrect_predictions_over_predunion_full_simulation_base_plus_sed_only.png",
    title_all: str = "F1 Across Full Simulation",
    title_base: str = "F1 Across Full Simulation: Base",
    title_sed: str = "F1 Across Full Simulation: Base + SED",
    title_bce_all: str = "BCE Across Full Simulation",
    title_bce_base: str = "BCE Across Full Simulation: Base",
    title_bce_sed: str = "BCE Across Full Simulation: Base + SED",
    title_incorrect_all: str = "Incorrect Predictions (FP + FN) Across Full Simulation",
    title_incorrect_base: str = "Incorrect Predictions (FP + FN) Across Full Simulation: Base",
    title_incorrect_sed: str = "Incorrect Predictions (FP + FN) Across Full Simulation: Base + SED",
    title_incorrect_norm_npixels_all: str = "Normalized Incorrect Predictions: (FP + FN) / n_pixels",
    title_incorrect_norm_npixels_base: str = "Normalized Incorrect Predictions: (FP + FN) / n_pixels, Base",
    title_incorrect_norm_npixels_sed: str = "Normalized Incorrect Predictions: (FP + FN) / n_pixels, Base + SED",
    title_incorrect_over_predunion_all: str = "Incorrect Predictions Relative to Predicted/True Positive Union: (FP + FN) / (TP + FP + FN)",
    title_incorrect_over_predunion_base: str = "Incorrect Predictions Relative to Predicted/True Positive Union: Base",
    title_incorrect_over_predunion_sed: str = "Incorrect Predictions Relative to Predicted/True Positive Union: Base + SED",
    individual_out_dir: Optional[str] = None,
    individual_bce_out_dir: Optional[str] = None,
    individual_incorrect_out_dir: Optional[str] = None,
    individual_incorrect_norm_npixels_out_dir: Optional[str] = None,
    individual_incorrect_over_predunion_out_dir: Optional[str] = None,
    frames_per_ns: Optional[float] = None,
    time_scale: float = 10.0,
    show: bool = True,
    linewidth: float = 2.0,
    colors_all: Optional[Sequence[str]] = None,
    colors_base: Optional[Sequence[str]] = None,
    colors_sed: Optional[Sequence[str]] = None,
    individual_color: Optional[str] = None,
    bce_colors_all: Optional[Sequence[str]] = None,
    bce_colors_base: Optional[Sequence[str]] = None,
    bce_colors_sed: Optional[Sequence[str]] = None,
    individual_bce_color: Optional[str] = None,
    incorrect_colors_all: Optional[Sequence[str]] = None,
    incorrect_colors_base: Optional[Sequence[str]] = None,
    incorrect_colors_sed: Optional[Sequence[str]] = None,
    individual_incorrect_color: Optional[str] = None,
    incorrect_norm_npixels_colors_all: Optional[Sequence[str]] = None,
    incorrect_norm_npixels_colors_base: Optional[Sequence[str]] = None,
    incorrect_norm_npixels_colors_sed: Optional[Sequence[str]] = None,
    individual_incorrect_norm_npixels_color: Optional[str] = None,
    incorrect_over_predunion_colors_all: Optional[Sequence[str]] = None,
    incorrect_over_predunion_colors_base: Optional[Sequence[str]] = None,
    incorrect_over_predunion_colors_sed: Optional[Sequence[str]] = None,
    individual_incorrect_over_predunion_color: Optional[str] = None,
    loss_colors: Optional[Sequence[str]] = None,
    loss_out_name: str = "training_and_validation_loss_comparison.png",
    val_loss_out_name: str = "training_and_validation_loss_comparison.png",
    loss_title: str = "Training and Validation Loss Comparison",
    val_loss_title: str = "Training and Validation Loss Comparison",
    loss_max_epoch: Optional[float] = None,
    make_f1_bar_at_time: bool = False,
    f1_bar_time_ns: Optional[float] = None,
    f1_bar_out_name: str = "f1_bar_comparison_at_time.png",
    f1_bar_title: Optional[str] = None,
    f1_bar_colors: Optional[Sequence[str]] = None,

    # New TP / FP / FN geometry bar plot controls
    make_confusion_bar_by_geometry: bool = False,
    confusion_bar_out_name: str = "tp_fp_fn_bar_by_geometry.png",
    confusion_bar_title: Optional[str] = None,
    confusion_bar_use_totals: bool = True,
    confusion_bar_time_ns: Optional[float] = None,
    confusion_bar_model_order: Optional[Sequence[str]] = None,
    confusion_bar_geometry_order: Optional[Sequence[str]] = None,
    confusion_bar_colors: Optional[Sequence[str]] = None,
    confusion_bar_xlabel: str = "Test geometry",

    # New F1 / Recall / Precision geometry bar plot controls
    make_classification_metrics_bar_by_geometry: bool = False,
    classification_metrics_bar_out_name: str = "f1_recall_precision_by_geometry.png",
    classification_metrics_bar_title: Optional[str] = None,
    classification_metrics_bar_use_totals: bool = True,
    classification_metrics_bar_time_ns: Optional[float] = None,
    classification_metrics_bar_model_order: Optional[Sequence[str]] = None,
    classification_metrics_bar_geometry_order: Optional[Sequence[str]] = None,
    classification_metrics_bar_colors: Optional[Sequence[str]] = None,
    classification_metrics_bar_xlabel: str = "Test geometry",
):
    """
    Make:
      1. F1 plots
      2. BCE plots
      3. Incorrect-prediction plots where incorrect_predictions = fp + fn
      4. Normalized incorrect-prediction plots where (fp + fn) / n_pixels
      5. Relative incorrect-prediction plots where (fp + fn) / (tp + fp + fn)
      6. Optional training/validation loss plots from training log CSVs
      7. Optional grouped F1 bar chart at a requested time_ns
      8. Optional TP/FP/FN grouped bar chart by test geometry
      9. Optional F1/Recall/Precision grouped bar chart by test geometry
    """
    os.makedirs(out_dir, exist_ok=True)

    if individual_out_dir is None:
        individual_out_dir = os.path.join(out_dir, "individual_f1_plots")
    os.makedirs(individual_out_dir, exist_ok=True)

    if individual_bce_out_dir is None:
        individual_bce_out_dir = os.path.join(out_dir, "individual_bce_plots")
    os.makedirs(individual_bce_out_dir, exist_ok=True)

    if individual_incorrect_out_dir is None:
        individual_incorrect_out_dir = os.path.join(out_dir, "individual_incorrect_prediction_plots")
    os.makedirs(individual_incorrect_out_dir, exist_ok=True)

    if individual_incorrect_norm_npixels_out_dir is None:
        individual_incorrect_norm_npixels_out_dir = os.path.join(
            out_dir, "individual_incorrect_prediction_per_npixels_plots"
        )
    os.makedirs(individual_incorrect_norm_npixels_out_dir, exist_ok=True)

    if individual_incorrect_over_predunion_out_dir is None:
        individual_incorrect_over_predunion_out_dir = os.path.join(
            out_dir, "individual_incorrect_prediction_over_predunion_plots"
        )
    os.makedirs(individual_incorrect_over_predunion_out_dir, exist_ok=True)

    save_all = os.path.join(out_dir, out_name_all)
    save_base = os.path.join(out_dir, out_name_base)
    save_sed = os.path.join(out_dir, out_name_sed)

    save_bce_all = os.path.join(out_dir, out_name_bce_all)
    save_bce_base = os.path.join(out_dir, out_name_bce_base)
    save_bce_sed = os.path.join(out_dir, out_name_bce_sed)

    save_incorrect_all = os.path.join(out_dir, out_name_incorrect_all)
    save_incorrect_base = os.path.join(out_dir, out_name_incorrect_base)
    save_incorrect_sed = os.path.join(out_dir, out_name_incorrect_sed)

    save_incorrect_norm_npixels_all = os.path.join(out_dir, out_name_incorrect_norm_npixels_all)
    save_incorrect_norm_npixels_base = os.path.join(out_dir, out_name_incorrect_norm_npixels_base)
    save_incorrect_norm_npixels_sed = os.path.join(out_dir, out_name_incorrect_norm_npixels_sed)

    save_incorrect_over_predunion_all = os.path.join(out_dir, out_name_incorrect_over_predunion_all)
    save_incorrect_over_predunion_base = os.path.join(out_dir, out_name_incorrect_over_predunion_base)
    save_incorrect_over_predunion_sed = os.path.join(out_dir, out_name_incorrect_over_predunion_sed)

    loaded_all = _plot_subset(
        csv_map,
        save_all,
        title=title_all,
        y_col="f1",
        y_label="F1",
        y_lim=(0.0, 1.05),
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        colors=colors_all,
    )

    base_map = {"Base": csv_map["Base"]} if "Base" in csv_map else {}
    loaded_base = []
    if base_map:
        loaded_base = _plot_subset(
            base_map,
            save_base,
            title=title_base,
            y_col="f1",
            y_label="F1",
            y_lim=(0.0, 1.05),
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=colors_base,
        )

    sed_map = {"Base + SED": csv_map["Base + SED"]} if "Base + SED" in csv_map else {}
    loaded_sed = []
    if sed_map:
        loaded_sed = _plot_subset(
            sed_map,
            save_sed,
            title=title_sed,
            y_col="f1",
            y_label="F1",
            y_lim=(0.0, 1.05),
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=colors_sed,
        )

    individual_plots = _plot_individual_simulations(
        csv_map,
        individual_out_dir,
        y_col="f1",
        y_label="F1",
        y_lim=(0.0, 1.05),
        start_from_positive_only=True,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        color=individual_color,
    )

    loaded_bce_all = _plot_subset(
        csv_map,
        save_bce_all,
        title=title_bce_all,
        y_col="bce",
        y_label="BCE",
        y_lim=None,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        colors=bce_colors_all if bce_colors_all is not None else colors_all,
    )

    loaded_bce_base = []
    if base_map:
        loaded_bce_base = _plot_subset(
            base_map,
            save_bce_base,
            title=title_bce_base,
            y_col="bce",
            y_label="BCE",
            y_lim=None,
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=bce_colors_base if bce_colors_base is not None else colors_base,
        )

    loaded_bce_sed = []
    if sed_map:
        loaded_bce_sed = _plot_subset(
            sed_map,
            save_bce_sed,
            title=title_bce_sed,
            y_col="bce",
            y_label="BCE",
            y_lim=None,
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=bce_colors_sed if bce_colors_sed is not None else colors_sed,
        )

    individual_bce_plots = _plot_individual_simulations(
        csv_map,
        individual_bce_out_dir,
        y_col="bce",
        y_label="BCE",
        y_lim=None,
        start_from_positive_only=False,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        color=individual_bce_color if individual_bce_color is not None else individual_color,
    )

    loaded_incorrect_all = _plot_subset(
        csv_map,
        save_incorrect_all,
        title=title_incorrect_all,
        y_col="incorrect_predictions",
        y_label="Incorrect predictions (FP + FN)",
        y_lim=None,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        colors=incorrect_colors_all if incorrect_colors_all is not None else colors_all,
    )

    loaded_incorrect_base = []
    if base_map:
        loaded_incorrect_base = _plot_subset(
            base_map,
            save_incorrect_base,
            title=title_incorrect_base,
            y_col="incorrect_predictions",
            y_label="Incorrect predictions (FP + FN)",
            y_lim=None,
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=incorrect_colors_base if incorrect_colors_base is not None else colors_base,
        )

    loaded_incorrect_sed = []
    if sed_map:
        loaded_incorrect_sed = _plot_subset(
            sed_map,
            save_incorrect_sed,
            title=title_incorrect_sed,
            y_col="incorrect_predictions",
            y_label="Incorrect predictions (FP + FN)",
            y_lim=None,
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=incorrect_colors_sed if incorrect_colors_sed is not None else colors_sed,
        )

    individual_incorrect_plots = _plot_individual_simulations(
        csv_map,
        individual_incorrect_out_dir,
        y_col="incorrect_predictions",
        y_label="Incorrect predictions (FP + FN)",
        y_lim=None,
        start_from_positive_only=False,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        color=individual_incorrect_color if individual_incorrect_color is not None else individual_color,
    )

    loaded_incorrect_norm_npixels_all = _plot_subset(
        csv_map,
        save_incorrect_norm_npixels_all,
        title=title_incorrect_norm_npixels_all,
        y_col="incorrect_predictions_per_npixels",
        y_label="Incorrect predictions / n_pixels",
        y_lim=(0.0, 0.3),
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        colors=(
            incorrect_norm_npixels_colors_all
            if incorrect_norm_npixels_colors_all is not None
            else colors_all
        ),
    )

    loaded_incorrect_norm_npixels_base = []
    if base_map:
        loaded_incorrect_norm_npixels_base = _plot_subset(
            base_map,
            save_incorrect_norm_npixels_base,
            title=title_incorrect_norm_npixels_base,
            y_col="incorrect_predictions_per_npixels",
            y_label="Incorrect predictions / n_pixels",
            y_lim=(0.0, 0.3),
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=(
                incorrect_norm_npixels_colors_base
                if incorrect_norm_npixels_colors_base is not None
                else colors_base
            ),
        )

    loaded_incorrect_norm_npixels_sed = []
    if sed_map:
        loaded_incorrect_norm_npixels_sed = _plot_subset(
            sed_map,
            save_incorrect_norm_npixels_sed,
            title=title_incorrect_norm_npixels_sed,
            y_col="incorrect_predictions_per_npixels",
            y_label="Incorrect predictions / n_pixels",
            y_lim=(0.0, 0.3),
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=(
                incorrect_norm_npixels_colors_sed
                if incorrect_norm_npixels_colors_sed is not None
                else colors_sed
            ),
        )

    individual_incorrect_norm_npixels_plots = _plot_individual_simulations(
        csv_map,
        individual_incorrect_norm_npixels_out_dir,
        y_col="incorrect_predictions_per_npixels",
        y_label="Incorrect predictions / n_pixels",
        y_lim=(0.0, 0.3),
        start_from_positive_only=False,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        color=(
            individual_incorrect_norm_npixels_color
            if individual_incorrect_norm_npixels_color is not None
            else individual_color
        ),
    )

    loaded_incorrect_over_predunion_all = _plot_subset(
        csv_map,
        save_incorrect_over_predunion_all,
        title=title_incorrect_over_predunion_all,
        y_col="incorrect_predictions_over_predunion",
        y_label="(FP + FN) / (TP + FP + FN)",
        y_lim=(0.0, 2.05),
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        colors=(
            incorrect_over_predunion_colors_all
            if incorrect_over_predunion_colors_all is not None
            else colors_all
        ),
    )

    loaded_incorrect_over_predunion_base = []
    if base_map:
        loaded_incorrect_over_predunion_base = _plot_subset(
            base_map,
            save_incorrect_over_predunion_base,
            title=title_incorrect_over_predunion_base,
            y_col="incorrect_predictions_over_predunion",
            y_label="(FP + FN) / (TP + FP + FN)",
            y_lim=(0.0, 2.05),
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=(
                incorrect_over_predunion_colors_base
                if incorrect_over_predunion_colors_base is not None
                else colors_base
            ),
        )

    loaded_incorrect_over_predunion_sed = []
    if sed_map:
        loaded_incorrect_over_predunion_sed = _plot_subset(
            sed_map,
            save_incorrect_over_predunion_sed,
            title=title_incorrect_over_predunion_sed,
            y_col="incorrect_predictions_over_predunion",
            y_label="(FP + FN) / (TP + FP + FN)",
            y_lim=(0.0, 2.05),
            frames_per_ns=frames_per_ns,
            time_scale=time_scale,
            show=show,
            linewidth=linewidth,
            colors=(
                incorrect_over_predunion_colors_sed
                if incorrect_over_predunion_colors_sed is not None
                else colors_sed
            ),
        )

    individual_incorrect_over_predunion_plots = _plot_individual_simulations(
        csv_map,
        individual_incorrect_over_predunion_out_dir,
        y_col="incorrect_predictions_over_predunion",
        y_label="(FP + FN) / (TP + FP + FN)",
        y_lim=(0.0, 2.05),
        start_from_positive_only=False,
        frames_per_ns=frames_per_ns,
        time_scale=time_scale,
        show=show,
        linewidth=linewidth,
        color=(
            individual_incorrect_over_predunion_color
            if individual_incorrect_over_predunion_color is not None
            else individual_color
        ),
    )

    loss_plot_info = None
    if loss_csv_map is not None and len(loss_csv_map) > 0:
        loss_plot_info = plot_training_loss_comparison(
            loss_csv_map,
            out_dir,
            out_name_loss=loss_out_name,
            out_name_val_loss=val_loss_out_name,
            title_loss=loss_title,
            title_val_loss=val_loss_title,
            max_epoch=loss_max_epoch,
            show=show,
            linewidth=linewidth,
            colors=loss_colors,
        )

    f1_bar_info = None
    if make_f1_bar_at_time:
        if f1_bar_time_ns is None:
            raise ValueError("f1_bar_time_ns must be provided when make_f1_bar_at_time=True")

        f1_bar_info = plot_metric_bar_at_time(
            csv_map,
            out_dir,
            target_time_ns=f1_bar_time_ns,
            y_col="f1",
            out_name=f1_bar_out_name,
            title=f1_bar_title,
            y_label="F1",
            show=show,
            model_order=["Base", "Base + SED"],
            bar_colors=f1_bar_colors,
        )

    confusion_bar_info = None
    if make_confusion_bar_by_geometry:
        if not confusion_bar_use_totals and confusion_bar_time_ns is None:
            raise ValueError(
                "confusion_bar_time_ns must be provided when "
                "confusion_bar_use_totals=False"
            )

        confusion_bar_info = plot_confusion_counts_bar_by_geometry(
            csv_map,
            out_dir,
            out_name=confusion_bar_out_name,
            title=confusion_bar_title,
            show=show,
            model_order=confusion_bar_model_order,
            geometry_order=confusion_bar_geometry_order,
            use_totals=confusion_bar_use_totals,
            target_time_ns=confusion_bar_time_ns,
            colors=confusion_bar_colors,
            xlabel=confusion_bar_xlabel,
        )

    classification_metrics_bar_info = None
    if make_classification_metrics_bar_by_geometry:
        if (
            not classification_metrics_bar_use_totals
            and classification_metrics_bar_time_ns is None
        ):
            raise ValueError(
                "classification_metrics_bar_time_ns must be provided when "
                "classification_metrics_bar_use_totals=False"
            )

        classification_metrics_bar_info = plot_classification_metrics_bar_by_geometry(
            csv_map,
            out_dir,
            out_name=classification_metrics_bar_out_name,
            title=classification_metrics_bar_title,
            show=show,
            model_order=classification_metrics_bar_model_order,
            geometry_order=classification_metrics_bar_geometry_order,
            use_totals=classification_metrics_bar_use_totals,
            target_time_ns=classification_metrics_bar_time_ns,
            colors=classification_metrics_bar_colors,
            xlabel=classification_metrics_bar_xlabel,
        )

    return {
        "all_plot_path": save_all,
        "base_plot_path": save_base if base_map else None,
        "base_plus_sed_plot_path": save_sed if sed_map else None,
        "individual_plot_dir": individual_out_dir,
        "individual_plot_count": len(individual_plots),
        "individual_plot_info": individual_plots,
        "loaded_all_series": loaded_all,
        "loaded_base_series": loaded_base,
        "loaded_base_plus_sed_series": loaded_sed,

        "bce_all_plot_path": save_bce_all,
        "bce_base_plot_path": save_bce_base if base_map else None,
        "bce_base_plus_sed_plot_path": save_bce_sed if sed_map else None,
        "individual_bce_plot_dir": individual_bce_out_dir,
        "individual_bce_plot_count": len(individual_bce_plots),
        "individual_bce_plot_info": individual_bce_plots,
        "loaded_bce_all_series": loaded_bce_all,
        "loaded_bce_base_series": loaded_bce_base,
        "loaded_bce_base_plus_sed_series": loaded_bce_sed,

        "incorrect_all_plot_path": save_incorrect_all,
        "incorrect_base_plot_path": save_incorrect_base if base_map else None,
        "incorrect_base_plus_sed_plot_path": save_incorrect_sed if sed_map else None,
        "individual_incorrect_plot_dir": individual_incorrect_out_dir,
        "individual_incorrect_plot_count": len(individual_incorrect_plots),
        "individual_incorrect_plot_info": individual_incorrect_plots,
        "loaded_incorrect_all_series": loaded_incorrect_all,
        "loaded_incorrect_base_series": loaded_incorrect_base,
        "loaded_incorrect_base_plus_sed_series": loaded_incorrect_sed,

        "incorrect_norm_npixels_all_plot_path": save_incorrect_norm_npixels_all,
        "incorrect_norm_npixels_base_plot_path": (
            save_incorrect_norm_npixels_base if base_map else None
        ),
        "incorrect_norm_npixels_base_plus_sed_plot_path": (
            save_incorrect_norm_npixels_sed if sed_map else None
        ),
        "individual_incorrect_norm_npixels_plot_dir": individual_incorrect_norm_npixels_out_dir,
        "individual_incorrect_norm_npixels_plot_count": len(individual_incorrect_norm_npixels_plots),
        "individual_incorrect_norm_npixels_plot_info": individual_incorrect_norm_npixels_plots,
        "loaded_incorrect_norm_npixels_all_series": loaded_incorrect_norm_npixels_all,
        "loaded_incorrect_norm_npixels_base_series": loaded_incorrect_norm_npixels_base,
        "loaded_incorrect_norm_npixels_base_plus_sed_series": loaded_incorrect_norm_npixels_sed,

        "incorrect_over_predunion_all_plot_path": save_incorrect_over_predunion_all,
        "incorrect_over_predunion_base_plot_path": (
            save_incorrect_over_predunion_base if base_map else None
        ),
        "incorrect_over_predunion_base_plus_sed_plot_path": (
            save_incorrect_over_predunion_sed if sed_map else None
        ),
        "individual_incorrect_over_predunion_plot_dir": individual_incorrect_over_predunion_out_dir,
        "individual_incorrect_over_predunion_plot_count": len(individual_incorrect_over_predunion_plots),
        "individual_incorrect_over_predunion_plot_info": individual_incorrect_over_predunion_plots,
        "loaded_incorrect_over_predunion_all_series": loaded_incorrect_over_predunion_all,
        "loaded_incorrect_over_predunion_base_series": loaded_incorrect_over_predunion_base,
        "loaded_incorrect_over_predunion_base_plus_sed_series": loaded_incorrect_over_predunion_sed,

        "loss_plot_info": loss_plot_info,
        "training_loss_plot_path": None if loss_plot_info is None else loss_plot_info["training_loss_plot_path"],
        "validation_loss_plot_path": None if loss_plot_info is None else loss_plot_info["validation_loss_plot_path"],

        "f1_bar_info": f1_bar_info,
        "f1_bar_plot_path": None if f1_bar_info is None else f1_bar_info["bar_plot_path"],

        "confusion_bar_info": confusion_bar_info,
        "confusion_bar_plot_path": (
            None if confusion_bar_info is None
            else confusion_bar_info["confusion_bar_plot_path"]
        ),

        "classification_metrics_bar_info": classification_metrics_bar_info,
        "classification_metrics_bar_plot_path": (
            None if classification_metrics_bar_info is None
            else classification_metrics_bar_info["classification_metrics_bar_plot_path"]
        ),
    }