# source/import_datasets.py

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf


def csv_dataset_from_directory(
    directory,
    *,
    pattern="**/*.csv",
    batch_size=5,
    shuffle=False,
    # IMPORTANT: downstream expects "SED" as the canonical column name.
    # Files may contain either "SED" or legacy "a_pos"; we normalize to "SED".
    feature_cols=("fracture_mask", "x", "y", "ux", "uy", "Gc", "pressure", "vonmises", "SED"),
    target_col="fracture_mask",
    # --- autoregressive settings ---
    sequence_length=10,
    shift=1,
    # --- extra scalar channel ---
    add_velocity=True,
    velocity_scale=1.0 / 1000.0,
    ignore_bad_files=False,
    # --- cleanup + diagnostics ---
    drop_first_csv=True,
    print_run_stats=True,
    stats_max_files=None,
    dataset_name="dataset",
    # --- mask cleaning ---
    clip_mask_to_01=True,
    binarize_mask=False,
    mask_threshold=0.5,
    # --- velocity detection print ---
    print_detected_velocities=True,
    max_examples_detected=5,
):
    """
    Returns tf.data.Dataset yielding:
      X: (B, T, N, Cin)
      y: (B, T, N, 1)

    X is frames [t..t+T-1] built from `feature_cols` (+ optional velocity channel).
    y is `target_col` from frames [t+shift..t+shift+T-1].

    Expected folder structure (flexible):
      directory/<run_folder>/*.csv

    Velocity parsing:
      - Supports run folder names containing token like: _V100_  (e.g., F_MS201_V100_out)
      - velocity stored as float(100) * velocity_scale (e.g., 0.1 if scale=1/1000)

    Notes:
      - time_ns is intentionally ignored and never required.
      - SED normalization: accepts either "SED" or legacy "a_pos" and standardizes to "SED".
    """

    directory = str(directory)

    # -----------------------------
    # 1) Find CSV files
    # -----------------------------
    glob_pat = f"{directory}/{pattern}"
    all_paths = sorted(tf.io.gfile.glob(glob_pat))
    if len(all_paths) == 0:
        raise ValueError(f"No CSV files found under: {glob_pat}")
    print(f"[{dataset_name}] Found {len(all_paths)} CSV files under {directory}")

    # -----------------------------
    # 2) Helpers
    # -----------------------------
    _V_PAT = re.compile(r"(?:^|_)V(\d+(?:\.\d+)?)(?:_|$)")

    def _standardize_sed_inplace(df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept either 'SED' or 'a_pos' in the CSV, but ensure downstream code
        always sees a column named 'SED'.

        - If 'SED' exists: do nothing
        - Else if 'a_pos' exists: create df['SED'] = df['a_pos']
        - Else: do nothing (missing-col validation will raise later)
        """
        if "SED" in df.columns:
            return df
        if "a_pos" in df.columns:
            df = df.copy()
            df["SED"] = df["a_pos"]
        return df

    def _parse_velocity_from_run(run_name: str) -> float:
        """
        Extract velocity from run folder names like:
          F_MS201_V100_out -> 100
        then applies velocity_scale (e.g., 1/1000).
        """
        m = _V_PAT.search(run_name)
        if not m:
            raise ValueError(f"Cannot parse velocity from run folder name: {run_name}")
        return float(m.group(1)) * velocity_scale

    def _parse_frame_idx(path: str, run_folder: str) -> int:
        """
        Extract frame index from filename.

        Supports:
          <run_folder>_t0000_0.00ns.csv  -> 0
          <run_folder>_t0123.csv         -> 123
          <run_folder>_0007.csv          -> 7   (legacy)
          anything_0042.csv              -> 42  (fallback)
        """
        base = os.path.splitext(os.path.basename(path))[0]  # drop .csv

        # Remove the run_folder prefix if present
        rest = base
        if base.startswith(run_folder):
            rest = base[len(run_folder):]
            if rest.startswith("_"):
                rest = rest[1:]

        # 1) Preferred: parse `_t####` anywhere in the remainder
        m = re.search(r"(?:^|_)t(\d+)(?:_|$)", rest)
        if m:
            return int(m.group(1))

        # 2) Legacy: ends with `_####`
        m = re.search(r"(?:^|_)(\d+)$", rest)
        if m:
            return int(m.group(1))

        # 3) Last resort: look in the full base
        m = re.search(r"(?:^|_)t(\d+)(?:_|$)", base)
        if m:
            return int(m.group(1))

        raise ValueError(f"Cannot parse frame index from: {path}")

    # -----------------------------
    # 3) Group by run folder
    # -----------------------------
    runs: Dict[str, List[str]] = {}
    for p in all_paths:
        run_folder = os.path.basename(os.path.dirname(p))
        runs.setdefault(run_folder, []).append(p)

    # -----------------------------
    # 3b) Print detected velocities (no hard-coded folder names)
    # -----------------------------
    if add_velocity and print_detected_velocities:
        detected: Dict[str, float] = {}
        failed: List[str] = []
        for run_folder in sorted(runs.keys()):
            try:
                detected[run_folder] = _parse_velocity_from_run(run_folder)
            except ValueError:
                failed.append(run_folder)

        unique_vels = sorted(set(detected.values()))
        print(
            f"[{dataset_name}] Detected velocities (scaled) in '{directory}': {unique_vels} "
            f"(count={len(unique_vels)})"
        )
        for rf, vv in list(sorted(detected.items()))[: int(max_examples_detected)]:
            print(f"[{dataset_name}]   {rf} -> {vv}")

        if failed:
            print(
                f"[{dataset_name}] WARNING: {len(failed)} run folder(s) contained CSVs but did not match "
                f"velocity pattern '*_V###_*'. Examples: {failed[:3]}"
            )

    # -----------------------------
    # 4) Infer num_points + validate columns using first readable CSV
    # -----------------------------
    num_points = None
    first_good = None
    first_df = None

    for p in all_paths:
        try:
            tmp = pd.read_csv(p)
            tmp = _standardize_sed_inplace(tmp)  # <-- SED/a_pos normalization
            num_points = len(tmp)
            first_good = p
            first_df = tmp
            break
        except Exception:
            if not ignore_bad_files:
                raise

    if num_points is None:
        raise ValueError("Could not read any CSVs to infer num_points.")

    required_cols = set(feature_cols) | {target_col}
    missing = sorted(required_cols - set(first_df.columns))
    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing required columns in CSV.\n"
            f"  Missing: {missing}\n"
            f"  Example file: {first_good}\n"
            f"  Present columns: {list(first_df.columns)}\n"
            f"Note: time_ns is intentionally ignored and not required.\n"
            f"Also: for SED we accept either 'SED' or 'a_pos' (legacy)."
        )

    base_num_features = len(feature_cols)
    Cin = base_num_features + (1 if add_velocity else 0)

    feature_names = list(feature_cols)
    if add_velocity:
        feature_names.append("velocity")

    print(f"[{dataset_name}] NUM_POINTS: {num_points}  BASE_FEATURES: {base_num_features}  Cin: {Cin}")
    print(f"[{dataset_name}] First readable CSV: {first_good}")

    # -----------------------------
    # 5) Mask cleaning helper
    # -----------------------------
    def _clean_mask(arr: np.ndarray) -> np.ndarray:
        if clip_mask_to_01:
            arr = np.clip(arr, 0.0, 1.0)
        if binarize_mask:
            arr = (arr > mask_threshold).astype(np.float32)
        return arr

    # -----------------------------
    # 6) Optional stats
    # -----------------------------
    def _init_stats(nc: int) -> dict:
        return {
            "min": np.full((nc,), np.inf, dtype=np.float64),
            "max": np.full((nc,), -np.inf, dtype=np.float64),
            "sum": np.zeros((nc,), dtype=np.float64),
            "count": 0,
        }

    def _update_stats(stats: dict, arr2d: np.ndarray) -> None:
        arr2d = np.asarray(arr2d, dtype=np.float64)
        stats["min"] = np.minimum(stats["min"], arr2d.min(axis=0))
        stats["max"] = np.maximum(stats["max"], arr2d.max(axis=0))
        stats["sum"] += arr2d.sum(axis=0)
        stats["count"] += arr2d.shape[0]

    def _print_stats_table(run_folder: str, stats: dict, names: Sequence[str], prefix: str = "") -> None:
        means = stats["sum"] / max(stats["count"], 1)
        print(prefix + f"Run {run_folder} stats (min / max / mean):")
        for i, nm in enumerate(names):
            print(prefix + f"  {nm:>15}: {stats['min'][i]: .4e}  {stats['max'][i]: .4e}  {means[i]: .4e}")

    if print_run_stats:
        print(
            f"[{dataset_name}] Computing per-run stats... "
            f"(drop_first_csv={drop_first_csv}, stats_max_files={stats_max_files})"
        )
        for run_folder, paths in sorted(runs.items()):
            paths_sorted_full = sorted(paths, key=lambda p: _parse_frame_idx(p, run_folder))

            if drop_first_csv and len(paths_sorted_full) > 0:
                dropped_path = paths_sorted_full[0]
                paths_sorted = paths_sorted_full[1:]
                print(f"[{dataset_name}] {run_folder}: dropped 1 file (dropped={os.path.basename(dropped_path)})")
            else:
                paths_sorted = paths_sorted_full

            if len(paths_sorted) == 0:
                continue

            scan_paths = paths_sorted if stats_max_files is None else paths_sorted[: int(stats_max_files)]

            v = _parse_velocity_from_run(run_folder) if add_velocity else None

            Xstats = _init_stats(Cin)
            Ystats = _init_stats(1)

            ok = 0
            for p in scan_paths:
                try:
                    df = pd.read_csv(p)
                    df = _standardize_sed_inplace(df)  # <-- SED/a_pos normalization

                    if len(df) != num_points:
                        raise ValueError(f"{p} has {len(df)} rows, expected {num_points}")

                    feats = df[list(feature_cols)].to_numpy(dtype=np.float32)  # (N,F)

                    # If fracture_mask is one of the feature cols, optionally clean it in-place
                    if target_col in feature_cols:
                        j = list(feature_cols).index(target_col)
                        feats[:, j:j + 1] = _clean_mask(feats[:, j:j + 1])

                    if add_velocity:
                        vcol = np.full((num_points, 1), v, dtype=np.float32)
                        feats = np.concatenate([feats, vcol], axis=1)

                    _update_stats(Xstats, feats)

                    tgt = df[[target_col]].to_numpy(dtype=np.float32)
                    tgt = _clean_mask(tgt)
                    _update_stats(Ystats, tgt)

                    ok += 1
                except Exception:
                    if ignore_bad_files:
                        continue
                    raise

            if ok:
                prefix = f"[{dataset_name}] "
                _print_stats_table(run_folder, Xstats, feature_names, prefix=prefix)
                _print_stats_table(run_folder, Ystats, [target_col], prefix=prefix)

        print(f"[{dataset_name}] Done computing per-run stats.\n")

    # -----------------------------
    # 7) Build sliding windows
    # -----------------------------
    in_windows: List[List[str]] = []
    out_windows: List[List[str]] = []
    vels: List[float] = []

    T = int(sequence_length)
    S = int(shift)

    for run_folder, paths in runs.items():
        paths_sorted_full = sorted(paths, key=lambda p: _parse_frame_idx(p, run_folder))
        paths_sorted = paths_sorted_full[1:] if (drop_first_csv and len(paths_sorted_full) > 0) else paths_sorted_full

        if len(paths_sorted) < T + S:
            continue

        v = _parse_velocity_from_run(run_folder) if add_velocity else 0.0

        for start in range(0, len(paths_sorted) - (T + S) + 1):
            in_windows.append(paths_sorted[start : start + T])
            out_windows.append(paths_sorted[start + S : start + S + T])
            vels.append(v)

    if len(in_windows) == 0:
        raise ValueError("No valid sliding windows produced. Check folder structure and sequence_length/shift.")

    in_windows = np.asarray(in_windows, dtype=np.str_)
    out_windows = np.asarray(out_windows, dtype=np.str_)
    vels = np.asarray(vels, dtype=np.float32)

    print(f"[{dataset_name}] Built {len(in_windows)} sliding-window samples (T={T}, shift={S})")

    # -----------------------------
    # 8) Per-window reader
    # -----------------------------
    feature_cols_list = list(feature_cols)
    mask_idx = feature_cols_list.index(target_col) if target_col in feature_cols_list else None

    def _read_window_py(in_paths_bytes, out_paths_bytes, vel_scalar):
        in_paths = [p.decode("utf-8") for p in in_paths_bytes.tolist()]
        out_paths = [p.decode("utf-8") for p in out_paths_bytes.tolist()]
        v = float(vel_scalar)

        X_seq, y_seq = [], []

        for p in in_paths:
            df = pd.read_csv(p)
            df = _standardize_sed_inplace(df)  # <-- SED/a_pos normalization

            if len(df) != num_points:
                raise ValueError(f"{p} has {len(df)} rows, expected {num_points}")

            feats = df[feature_cols_list].to_numpy(dtype=np.float32)  # (N,F)

            # Clean fracture_mask channel inside X if present
            if mask_idx is not None:
                feats[:, mask_idx : mask_idx + 1] = _clean_mask(feats[:, mask_idx : mask_idx + 1])

            if add_velocity:
                vcol = np.full((num_points, 1), v, dtype=np.float32)
                feats = np.concatenate([feats, vcol], axis=1)

            X_seq.append(feats)

        for p in out_paths:
            df = pd.read_csv(p)
            df = _standardize_sed_inplace(df)  # <-- SED/a_pos normalization

            if len(df) != num_points:
                raise ValueError(f"{p} has {len(df)} rows, expected {num_points}")

            tgt = df[[target_col]].to_numpy(dtype=np.float32)
            tgt = _clean_mask(tgt)
            y_seq.append(tgt)

        X_seq = np.stack(X_seq, axis=0)  # (T,N,Cin)
        y_seq = np.stack(y_seq, axis=0)  # (T,N,1)
        return X_seq, y_seq

    def _read_window_tf(in_paths, out_paths, vel):
        X, y = tf.py_function(
            func=lambda ip, op, vv: _read_window_py(ip.numpy(), op.numpy(), vv.numpy()),
            inp=[in_paths, out_paths, vel],
            Tout=[tf.float32, tf.float32],
        )
        X.set_shape([T, num_points, Cin])
        y.set_shape([T, num_points, 1])
        return X, y

    # -----------------------------
    # 9) Build tf.data dataset
    # -----------------------------
    ds = tf.data.Dataset.from_tensor_slices((in_windows, out_windows, vels))

    if shuffle:
        ds = ds.shuffle(min(len(in_windows), 2000), reshuffle_each_iteration=True)

    ds = ds.map(_read_window_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if ignore_bad_files:
        ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds