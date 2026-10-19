from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_names: list[str],
) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(variable_names):
        truth = y_true[:, idx].astype(np.float64)
        pred = y_pred[:, idx].astype(np.float64)
        mask = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(mask):
            rows.append(
                {
                    "variable": name,
                    "n": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                    "r2": np.nan,
                    "pearson": np.nan,
                }
            )
            continue
        truth = truth[mask]
        pred = pred[mask]
        err = pred - truth
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        denom = float(np.sum((truth - np.mean(truth)) ** 2))
        r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else np.nan
        if truth.size >= 2 and np.std(truth) > 0 and np.std(pred) > 0:
            pearson = float(np.corrcoef(truth, pred)[0, 1])
        else:
            pearson = np.nan
        rows.append(
            {
                "variable": name,
                "n": int(truth.size),
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "r2": r2,
                "pearson": pearson,
            }
        )
    return pd.DataFrame(rows)
