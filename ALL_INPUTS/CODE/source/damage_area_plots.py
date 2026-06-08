"""
damage_area_plots.py

Utilities for calculating and plotting damaged area over time from ML prediction
arrays and optional FEM CSV output.

Typical use in a notebook:

    import importlib
    import source.damage_area_plots as dap
    importlib.reload(dap)

    results = dap.plot_damaged_area_over_time_ml_fem(...)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def _natural_sort_key(path: Path):
    """Sort filenames like frame_2.npy before frame_10.npy."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _to_2d_mask(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a saved prediction array into a 2D binary damaged mask.

    Works for common saved shapes:
      (H, W)
      (H, W, 1)
      (1, H, W)
      (1, H, W, 1)

    Returns
    -------
    damaged_mask : np.ndarray of bool, shape (H, W)
        True where the predicted damage value is >= threshold.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask after squeeze, but got shape {arr.shape}. "
            "Check the saved .npy array shape."
        )

    return arr >= threshold


def calculate_area_per_element_um2(element_size_nm: float = 781.25) -> float:
    """
    Convert square element side length from nm to area in micrometers squared.

    area_um2 = element_size_nm^2 / 1e6

    because:
      1 micrometer = 1000 nm
      1 micrometer^2 = 1,000,000 nm^2
    """
    return (element_size_nm ** 2) / 1_000_000.0


def calculate_ml_damaged_area_over_time(
    pred_arrays_dir: PathLike,
    *,
    max_ns: float = 25.0,
    frames_per_ns: float = 10.0,
    threshold: float = 0.5,
    element_size_nm: float = 781.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ML predicted damaged area over time from saved .npy arrays.

    Damaged area is calculated as:

        damaged_pixels = number of pixels where prediction >= threshold
        damaged_area_um2 = damaged_pixels * area_per_element_um2

    where:

        area_per_element_um2 = element_size_nm^2 / 1e6

    Returns
    -------
    times_ns : np.ndarray
        Time values in ns.
    damaged_area_um2 : np.ndarray
        Damaged area values in micrometers squared.
    damaged_pixel_counts : np.ndarray
        Raw damaged pixel/element counts before area scaling.
    """
    pred_arrays_dir = Path(pred_arrays_dir)

    npy_files = sorted(pred_arrays_dir.glob("*.npy"), key=_natural_sort_key)

    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in: {pred_arrays_dir}")

    max_frames = int(max_ns * frames_per_ns) + 1
    npy_files = npy_files[:max_frames]

    area_per_element_um2 = calculate_area_per_element_um2(
        element_size_nm=element_size_nm
    )

    times_ns = []
    damaged_pixel_counts = []
    damaged_area_um2 = []

    for i, f in enumerate(npy_files):
        arr = np.load(f)
        damaged_mask = _to_2d_mask(arr, threshold=threshold)

        damaged_pixels = np.sum(damaged_mask)
        damaged_area = damaged_pixels * area_per_element_um2
        time_ns = i / frames_per_ns

        times_ns.append(time_ns)
        damaged_pixel_counts.append(damaged_pixels)
        damaged_area_um2.append(damaged_area)

    return (
        np.asarray(times_ns),
        np.asarray(damaged_area_um2),
        np.asarray(damaged_pixel_counts),
    )


def load_fem_damaged_area_over_time(
    fem_csv_path: PathLike,
    *,
    max_ns: float = 25.0,
    time_col: str = "time_ns",
    damage_col: str = "elem_count",
    fem_damage_is_element_count: bool = True,
    element_size_nm: float = 781.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FEM damaged quantity over time from output.csv.

    If fem_damage_is_element_count=True:
        FEM damage_col is assumed to be damaged element count and is converted
        to damaged area in micrometers squared.

    If fem_damage_is_element_count=False:
        FEM damage_col is assumed to already be an area value.
    """
    fem_csv_path = Path(fem_csv_path)

    if not fem_csv_path.exists():
        raise FileNotFoundError(f"Could not find FEM CSV: {fem_csv_path}")

    df = pd.read_csv(fem_csv_path)

    if time_col not in df.columns:
        raise KeyError(
            f"Column '{time_col}' not found in {fem_csv_path.name}. "
            f"Available columns are: {list(df.columns)}"
        )

    if damage_col not in df.columns:
        raise KeyError(
            f"Column '{damage_col}' not found in {fem_csv_path.name}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df[[time_col, damage_col]].copy()
    df = df.dropna()
    df = df.sort_values(time_col)
    df = df[df[time_col] <= max_ns]

    fem_damage_values = df[damage_col].to_numpy()

    if fem_damage_is_element_count:
        area_per_element_um2 = calculate_area_per_element_um2(
            element_size_nm=element_size_nm
        )
        fem_damage_values = fem_damage_values * area_per_element_um2

    return df[time_col].to_numpy(), fem_damage_values


def _format_plot(
    *,
    font_size: float = 18,
    show_title: bool = True,
    title: str = "Damaged Area Over Time",
    x_label: str = "Time (ns)",
    y_label: str = r"Damaged Area ($\mu$m$^2$)",
    max_ns: float = 25.0,
    legend: bool = True,
    legend_font_size: Optional[float] = None,
    legend_loc: str = "center left",
    legend_bbox_to_anchor=(1.02, 0.5),
) -> None:
    """Apply common plot formatting."""
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    plt.xticks(fontsize=font_size * 0.85)
    plt.yticks(fontsize=font_size * 0.85)

    if show_title:
        plt.title(title, fontsize=font_size)

    plt.xlim(0, max_ns)

    if legend:
        if legend_font_size is None:
            legend_font_size = font_size * 0.8

        plt.legend(
            fontsize=legend_font_size,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=False,
        )

    plt.tight_layout()


def plot_damaged_area_over_time_ml_fem(
    pred_arrays_dir: PathLike,
    fem_csv_path: Optional[PathLike] = None,
    *,
    max_ns: float = 25.0,
    frames_per_ns: float = 10.0,
    threshold: float = 0.5,
    # Element scaling
    element_size_nm: float = 781.25,
    # Column names in FEM CSV
    fem_time_col: str = "time_ns",
    fem_damage_col: str = "elem_count",
    fem_damage_is_element_count: bool = True,
    # Colors and line settings
    ml_color: str = "dodgerblue",
    fem_color: str = "black",
    ml_label: str = "ML Prediction",
    fem_label: str = "FEM",
    linewidth: float = 2.5,
    fem_linewidth: Optional[float] = None,
    # Plot formatting
    font_size: float = 18,
    show_title: bool = True,
    title: str = "Damaged Area Over Time",
    x_label: str = "Time (ns)",
    y_label: str = r"Damaged Area ($\mu$m$^2$)",
    figsize=(9, 6),
    # Legend
    legend: bool = True,
    legend_font_size: Optional[float] = None,
    legend_loc: str = "center left",
    legend_bbox_to_anchor=(1.02, 0.5),
    # Save paths
    save_combined_path: Optional[PathLike] = None,
    save_ml_path: Optional[PathLike] = None,
    save_fem_path: Optional[PathLike] = None,
    # Display
    show: bool = True,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Calculate and plot damaged area over time for ML predictions and optional FEM data.

    ML damaged area is calculated by:
      1. thresholding each prediction array
      2. counting pixels where prediction >= threshold
      3. multiplying by element area in micrometers squared

    If fem_damage_is_element_count=True, FEM values are also multiplied by the
    same element area.
    """
    if fem_linewidth is None:
        fem_linewidth = linewidth

    area_per_element_um2 = calculate_area_per_element_um2(
        element_size_nm=element_size_nm
    )

    print(f"Element size: {element_size_nm} nm")
    print(f"Area per element: {area_per_element_um2:.10f} um^2")

    # ------------------------------------------------------------
    # Calculate ML damaged area
    # ------------------------------------------------------------
    ml_times_ns, ml_damaged_area_um2, ml_damaged_pixels = calculate_ml_damaged_area_over_time(
        pred_arrays_dir,
        max_ns=max_ns,
        frames_per_ns=frames_per_ns,
        threshold=threshold,
        element_size_nm=element_size_nm,
    )

    # ------------------------------------------------------------
    # Load FEM data, if provided
    # ------------------------------------------------------------
    fem_times_ns = None
    fem_damage_area_um2 = None

    if fem_csv_path is not None:
        fem_times_ns, fem_damage_area_um2 = load_fem_damaged_area_over_time(
            fem_csv_path,
            max_ns=max_ns,
            time_col=fem_time_col,
            damage_col=fem_damage_col,
            fem_damage_is_element_count=fem_damage_is_element_count,
            element_size_nm=element_size_nm,
        )

    # ------------------------------------------------------------
    # Combined ML + FEM plot
    # ------------------------------------------------------------
    if save_combined_path is not None or show:
        plt.figure(figsize=figsize)

        plt.plot(
            ml_times_ns,
            ml_damaged_area_um2,
            color=ml_color,
            linewidth=linewidth,
            label=ml_label,
        )

        if fem_times_ns is not None:
            plt.plot(
                fem_times_ns,
                fem_damage_area_um2,
                color=fem_color,
                linewidth=fem_linewidth,
                label=fem_label,
            )

        _format_plot(
            font_size=font_size,
            show_title=show_title,
            title=title,
            x_label=x_label,
            y_label=y_label,
            max_ns=max_ns,
            legend=legend,
            legend_font_size=legend_font_size,
            legend_loc=legend_loc,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

        if save_combined_path is not None:
            save_combined_path = Path(save_combined_path)
            save_combined_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_combined_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved combined ML + FEM plot to: {save_combined_path}")

        if show:
            plt.show()
        else:
            plt.close()

    # ------------------------------------------------------------
    # ML-only plot
    # ------------------------------------------------------------
    if save_ml_path is not None:
        plt.figure(figsize=figsize)

        plt.plot(
            ml_times_ns,
            ml_damaged_area_um2,
            color=ml_color,
            linewidth=linewidth,
            label=ml_label,
        )

        _format_plot(
            font_size=font_size,
            show_title=show_title,
            title=f"{title} - ML",
            x_label=x_label,
            y_label=y_label,
            max_ns=max_ns,
            legend=legend,
            legend_font_size=legend_font_size,
            legend_loc=legend_loc,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

        save_ml_path = Path(save_ml_path)
        save_ml_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_ml_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved ML-only plot to: {save_ml_path}")

    # ------------------------------------------------------------
    # FEM-only plot
    # ------------------------------------------------------------
    if save_fem_path is not None and fem_times_ns is not None:
        plt.figure(figsize=figsize)

        plt.plot(
            fem_times_ns,
            fem_damage_area_um2,
            color=fem_color,
            linewidth=fem_linewidth,
            label=fem_label,
        )

        _format_plot(
            font_size=font_size,
            show_title=show_title,
            title=f"{title} - FEM",
            x_label=x_label,
            y_label=y_label,
            max_ns=max_ns,
            legend=legend,
            legend_font_size=legend_font_size,
            legend_loc=legend_loc,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

        save_fem_path = Path(save_fem_path)
        save_fem_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fem_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved FEM-only plot to: {save_fem_path}")

    return {
        "ml_times_ns": ml_times_ns,
        "ml_damaged_area_um2": ml_damaged_area_um2,
        "ml_damaged_pixels": ml_damaged_pixels,
        "fem_times_ns": fem_times_ns,
        "fem_damaged_area_um2": fem_damage_area_um2,
        "area_per_element_um2": area_per_element_um2,
    }


__all__ = [
    "calculate_area_per_element_um2",
    "calculate_ml_damaged_area_over_time",
    "load_fem_damaged_area_over_time",
    "plot_damaged_area_over_time_ml_fem",
]
