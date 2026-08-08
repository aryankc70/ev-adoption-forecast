"""Model-agnostic evaluation metrics for EV adoption forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    """Aggregate and per-group evaluation metrics for a set of predictions."""

    mape: float
    rmse: float
    mae: float
    n_observations: int
    per_group_metrics: pd.DataFrame


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Uses actual + 1 in the denominator (a common MAPE variant) to avoid
    division-by-zero blowups when true share is 0% or very close to it,
    which happens often in our data (many countries start at ~0% share).
    """
    return float(np.mean(np.abs((actual - predicted) / (actual + 1.0))) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def evaluate_predictions(
    df: pd.DataFrame,
    actual_col: str = "ev_sales_share",
    predicted_col: str = "predicted_share",
    group_cols: list[str] | None = None,
) -> EvaluationResult:
    """
    Compute MAPE, RMSE, and MAE overall and per group (e.g. per country/mode).

    group_cols defaults to ["region_country", "mode"] -- per-group breakdown
    matters because a model can look good in aggregate while being terrible
    for specific countries (e.g. nailing large stable markets but missing
    fast-moving ones like Nepal or China).
    """
    if group_cols is None:
        group_cols = ["region_country", "mode"]

    actual = df[actual_col].to_numpy()
    predicted = df[predicted_col].to_numpy()

    overall_mape = _mape(actual, predicted)
    overall_rmse = _rmse(actual, predicted)
    overall_mae = _mae(actual, predicted)

    per_group_rows: list[dict[str, object]] = []
    for group_key, group_df in df.groupby(group_cols):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = dict(zip(group_cols, keys, strict=True))
        group_actual = group_df[actual_col].to_numpy()
        group_predicted = group_df[predicted_col].to_numpy()
        row["mape"] = _mape(group_actual, group_predicted)
        row["rmse"] = _rmse(group_actual, group_predicted)
        row["mae"] = _mae(group_actual, group_predicted)
        row["n"] = len(group_df)
        per_group_rows.append(row)

    per_group = pd.DataFrame(per_group_rows)

    return EvaluationResult(
        mape=overall_mape,
        rmse=overall_rmse,
        mae=overall_mae,
        n_observations=len(df),
        per_group_metrics=per_group,
    )
