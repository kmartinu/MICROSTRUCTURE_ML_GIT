# source/predict_full_simulation_metrics.py

from __future__ import annotations

import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def metrics_from_binary(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    """
    Compute binary classification metrics + confusion counts from flattened 0/1 arrays.
    Returns: (precision, recall, f1, accuracy, tp, tn, fp, fn)
    """
    cm = confusion_matrix(y_true_1d, y_pred_1d, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return prec, rec, f1, acc, tp, tn, fp, fn


def compute_bce_per_frame(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Mean binary cross entropy over all pixels in one frame.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth mask with values 0 or 1.
    y_prob : np.ndarray
        Predicted probabilities in [0, 1], same shape as y_true.
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    float
        Mean BCE over all pixels.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    bce = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(bce))


def plot_metric_over_time(
    df: pd.DataFrame,
    save_path: str,
    *,
    y_col: str,
    y_label: str,
    frames_per_ns: float | None = None,
    x_max: float | None = 70.0,
    linewidth: float = 2.0,
    show: bool = False,
):
    """
    Plot one metric over time from the saved dataframe.
    """
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in dataframe.")

    if "time_ns" in df.columns:
        x = pd.to_numeric(df["time_ns"], errors="coerce").to_numpy()
        x_label = "Time (ns)"
    else:
        x = pd.to_numeric(df["frame_id"], errors="coerce").to_numpy()
        if frames_per_ns is not None:
            x = x / float(frames_per_ns)
            x_label = "Time (ns)"
        else:
            x_label = "Frame ID"

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    plt.figure(figsize=(11, 6))
    plt.plot(x, y, linewidth=linewidth)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)

    if x_max is not None:
        plt.xlim(0, x_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved plot to: {save_path}")


def _get_model_expected_channels(model) -> int:
    """
    Infer the model's expected input channel count from model.input_shape.

    Supports:
      - single-input models with shape (None, T, H, W, C)
      - list-valued input_shape for multi-input models (uses the first input)

    Returns
    -------
    int
        Expected channel count C.
    """
    input_shape = getattr(model, "input_shape", None)

    if input_shape is None:
        raise ValueError("Could not read model.input_shape from the model.")

    if isinstance(input_shape, tuple):
        if len(input_shape) < 5:
            raise ValueError(f"Unexpected model.input_shape={input_shape}")
        c = input_shape[-1]
        if c is None:
            raise ValueError(f"Could not infer channel count from model.input_shape={input_shape}")
        return int(c)

    if isinstance(input_shape, list) and len(input_shape) > 0:
        first_shape = input_shape[0]
        if not isinstance(first_shape, tuple) or len(first_shape) < 5:
            raise ValueError(f"Unexpected first model input shape={first_shape}")
        c = first_shape[-1]
        if c is None:
            raise ValueError(f"Could not infer channel count from first model input shape={first_shape}")
        return int(c)

    raise ValueError(f"Unsupported model.input_shape format: {input_shape}")


def _adapt_channels_to_model(
    X: np.ndarray,
    model_expected_channels: int,
    fracture_channel_idx: int,
):
    """
    Adapt dataset channels to what the model expects.

    Rules
    -----
    - If dataset has exactly the same number of channels as the model expects:
        keep as-is
    - If dataset has MORE channels than the model expects:
        drop the highest-index channels until the counts match
        (assumes most recently added features are appended at the end)
    - If dataset has FEWER channels than the model expects:
        raise an error
    """
    if X.ndim != 5:
        raise ValueError(f"Expected X to have shape (B,T,H,W,C), got {X.shape}")

    cin = X.shape[-1]

    if cin == model_expected_channels:
        return X, fracture_channel_idx

    if cin < model_expected_channels:
        raise ValueError(
            f"Dataset has fewer channels than the model expects: "
            f"Cin={cin}, model_expected_channels={model_expected_channels}. "
            f"This code only auto-drops extra trailing channels; it does not invent missing ones."
        )

    keep_channels = list(range(model_expected_channels))
    dropped_channels = list(range(model_expected_channels, cin))

    X = X[..., keep_channels]

    if fracture_channel_idx in dropped_channels:
        raise ValueError(
            f"fracture_channel_idx={fracture_channel_idx} would be dropped when adapting "
            f"Cin={cin} to model_expected_channels={model_expected_channels}. "
            f"Check your fracture_channel_idx and channel ordering."
        )

    return X, fracture_channel_idx


def _prepare_x_window(
    X_tensor,
    model_expected_channels: int,
    fracture_channel_idx: int,
):
    """
    Convert tensor to numpy float32 and adapt channels to the model.
    """
    X_np = X_tensor.numpy().astype(np.float32)
    X_np, fracture_channel_idx_eff = _adapt_channels_to_model(
        X_np,
        model_expected_channels=model_expected_channels,
        fracture_channel_idx=fracture_channel_idx,
    )
    return X_np, fracture_channel_idx_eff


def predict_full_simulation_continuous_ar_with_metrics(
    ds_img,
    model,
    out_dir: str,
    *,
    threshold: float | None = 0.5,
    max_batches: Optional[int] = None,
    prefix: str = "mask",
    csv_path: Optional[str] = None,
    gt_threshold: float = 0.5,
    frames_per_ns: float | None = None,
    fracture_channel_idx: int = 0,
    max_ar_steps: Optional[int] = None,
    enforce_no_healing: bool = True,
    make_bce_plot: bool = True,
    bce_plot_path: Optional[str] = None,
):
    """
    Run a SINGLE continuous autoregressive rollout through ds_img.

    Behavior
    --------
    - Uses the first window in ds_img as the initial true input state.
    - For each subsequent step:
        * model predicts fracture from current window
        * compare predicted last frame against GT last frame of the matching dataset window
        * shift the input window causally (NO wraparound)
        * replace only the fracture channel in the new last frame with the prediction
        * use the true future values of all non-fracture channels from the next dataset window

    Channel handling
    ----------------
    - If test input has more channels than the model expects, drop the trailing
      highest-index channels until the counts match.

    No-healing behavior
    -------------------
    - If enforce_no_healing=True:
        once a pixel is predicted as fractured (1), it stays fractured forever
        for save output, metric evaluation, and AR feedback.

    BCE behavior
    ------------
    - BCE is computed from the raw model probabilities for the current frame:
          pred_last_raw
      not from the thresholded / no-healing binary mask.
    - Binary metrics (TP/TN/FP/FN/F1/etc.) still use the thresholded prediction
      after optional no-healing enforcement.

    Notes
    -----
    - The model predicts fracture only.
    - All non-fracture channels are treated as known exogenous inputs.
    - ds_img should be chronological and shuffle=False.
    - If the final dataset batch is smaller than the current AR batch, rollout stops
      gracefully and still saves/prints metrics accumulated so far.
    """
    os.makedirs(out_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(out_dir, "per_frame_metrics.csv")

    if bce_plot_path is None:
        bce_plot_path = os.path.join(out_dir, "bce_over_time.png")

    pred_threshold = 0.5 if threshold is None else float(threshold)

    t0 = time.time()
    rows = []
    frame_counter = 0
    TP = TN = FP = FN = 0

    model_expected_channels = _get_model_expected_channels(model)

    iterator = iter(ds_img.take(max_batches) if max_batches is not None else ds_img)

    # ---------------------------
    # Initialize from first window
    # ---------------------------
    try:
        X0, y0 = next(iterator)
    except StopIteration:
        raise ValueError("ds_img is empty; no windows available for continuous AR rollout.")

    X_ar, fracture_channel_idx_eff = _prepare_x_window(
        X0,
        model_expected_channels=model_expected_channels,
        fracture_channel_idx=fracture_channel_idx,
    )
    y0_np = y0.numpy().astype(np.float32)

    if X_ar.ndim != 5:
        raise ValueError(f"Expected X_ar shape (B,T,H,W,C), got {X_ar.shape}")
    if y0_np.ndim != 5:
        raise ValueError(f"Expected y0 shape (B,T,H,W,1), got {y0_np.shape}")

    B, T, H, W, Cin = X_ar.shape

    if fracture_channel_idx_eff >= Cin:
        raise ValueError(
            f"fracture_channel_idx={fracture_channel_idx_eff} out of range for Cin={Cin}"
        )

    print(f"fracture_channel_idx_eff={fracture_channel_idx_eff}")
    print(f"enforce_no_healing={enforce_no_healing}")

    pbar = tqdm(desc=f"Continuous AR {out_dir}", total=None)

    current_y_gt_last = y0_np[:, -1, :, :, 0]  # (B,H,W)

    running_fracture_state = X_ar[:, -1, :, :, fracture_channel_idx_eff].copy()
    running_fracture_state = (running_fracture_state >= pred_threshold).astype(np.float32)

    step_idx = 0

    while True:
        y_prob = model.predict(X_ar, verbose=0)

        if y_prob.ndim == 5:
            pred_last_raw = y_prob[:, -1, :, :, 0]   # (B,H,W)
        elif y_prob.ndim == 4:
            pred_last_raw = y_prob[:, :, :, 0]       # (B,H,W)
        else:
            raise ValueError(
                f"Unexpected model output shape {y_prob.shape}. "
                f"Expected 5D (B,T,H,W,1) or 4D (B,H,W,1)."
            )

        pred_last_bin = (pred_last_raw >= pred_threshold).astype(np.float32)

        if enforce_no_healing:
            pred_last_eval = np.maximum(running_fracture_state, pred_last_bin).astype(np.float32)
        else:
            pred_last_eval = pred_last_bin.astype(np.float32)

        # Evaluate against current GT
        for i in range(B):
            pred_mask = pred_last_eval[i]   # binary/no-heal mask for metrics + saving
            pred_prob = pred_last_raw[i]    # raw probability for BCE
            gt = current_y_gt_last[i]

            pred_flip = np.flipud(pred_mask)
            pred_prob_flip = np.flipud(pred_prob)
            gt_flip = np.flipud(gt)

            # Save PNG (binary mask)
            save_arr = pred_flip.astype(np.float32)
            img_u8 = (np.clip(save_arr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_u8).save(
                os.path.join(out_dir, f"{prefix}_{frame_counter:06d}.png")
            )

            # Binary metrics
            pred_bin = (pred_flip >= 0.5).astype(np.uint8)
            gt_bin = (gt_flip >= gt_threshold).astype(np.uint8)

            yt = gt_bin.reshape(-1)
            yp = pred_bin.reshape(-1)

            prec, rec, f1, acc, tp, tn, fp, fn = metrics_from_binary(yt, yp)
            bce = compute_bce_per_frame(gt_flip, pred_prob_flip)

            TP += tp
            TN += tn
            FP += fp
            FN += fn

            row = {
                "frame_id": frame_counter,
                "rollout_step": step_idx,
                "batch_source": "continuous_ar",
                "i_in_batch": i,
                "H": H,
                "W": W,
                "n_pixels": int(yt.size),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
                "bce": float(bce),
            }
            if frames_per_ns is not None:
                row["time_ns"] = frame_counter / float(frames_per_ns)

            rows.append(row)
            frame_counter += 1

        pbar.update(1)

        running_fracture_state = pred_last_eval.copy()

        if max_ar_steps is not None and step_idx >= max_ar_steps:
            break

        try:
            X_next_true_tensor, y_next_true_tensor = next(iterator)
        except StopIteration:
            break

        X_next_true, fracture_channel_idx_eff_next = _prepare_x_window(
            X_next_true_tensor,
            model_expected_channels=model_expected_channels,
            fracture_channel_idx=fracture_channel_idx,
        )
        y_next_true = y_next_true_tensor.numpy().astype(np.float32)

        if fracture_channel_idx_eff_next != fracture_channel_idx_eff:
            raise ValueError(
                "Fracture channel index changed unexpectedly after channel processing."
            )

        if X_next_true.shape != X_ar.shape:
            if X_next_true.shape[1:] == X_ar.shape[1:]:
                print(
                    "\nStopping rollout at dataset end because the next batch is smaller "
                    f"({X_next_true.shape[0]} instead of {X_ar.shape[0]}). "
                    "Metrics up to this point will still be saved."
                )
                break

            raise ValueError(
                f"Shape mismatch between current AR state {X_ar.shape} "
                f"and next true window {X_next_true.shape}"
            )

        next_mask = pred_last_eval[..., None].astype(np.float32)  # (B,H,W,1)

        X_ar_new = np.empty_like(X_ar)
        X_ar_new[:, :-1, :, :, :] = X_ar[:, 1:, :, :, :]
        X_ar_new[:, -1, :, :, :] = X_next_true[:, -1, :, :, :]
        X_ar_new[:, -1, :, :, fracture_channel_idx_eff:fracture_channel_idx_eff + 1] = next_mask

        X_ar = X_ar_new
        current_y_gt_last = y_next_true[:, -1, :, :, 0]
        step_idx += 1

    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    if make_bce_plot and len(df) > 0 and "bce" in df.columns:
        plot_metric_over_time(
            df,
            bce_plot_path,
            y_col="bce",
            y_label="BCE loss",
            frames_per_ns=frames_per_ns,
            x_max=70.0,
            linewidth=2.0,
            show=False,
        )

    total_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    total_rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    total_f1 = (2 * total_prec * total_rec / (total_prec + total_rec)) if (total_prec + total_rec) > 0 else 0.0
    total_acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    t1 = time.time()
    print(f"\nSaved {frame_counter} frames to: {out_dir}")
    print(f"Saved per-frame metrics CSV to: {csv_path}")
    if make_bce_plot:
        print(f"Saved BCE plot to: {bce_plot_path}")
    print("\nTOTAL METRICS")
    print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
    print(f"Precision: {total_prec:.6f}")
    print(f"Recall:    {total_rec:.6f}")
    print(f"F1:        {total_f1:.6f}")
    print(f"Accuracy:  {total_acc:.6f}")
    print(f"Elapsed time: {t1 - t0:.3f} s")

    return df