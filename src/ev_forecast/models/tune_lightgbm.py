"""Time-series cross-validated hyperparameter tuning for the LightGBM candidate."""

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

from ev_forecast.models.evaluation import evaluate_predictions
from ev_forecast.models.gbm_features import prepare_gbm_features
from ev_forecast.models.gbm_models import ModelCandidate
from ev_forecast.models.recursive_forecast import recursive_forecast

RANDOM_SEED = 42

# Expanding-window CV cutoffs: each fold trains on years <= cutoff and validates
# on the next two years, all strictly within the training period (<=2021) so the
# true 2022-2025 test set stays genuinely held out until final model selection.
CV_CUTOFFS = [2018, 2019]
CV_HORIZON_YEARS = 2

PARAM_GRID: dict[str, list[float]] = {
    "n_estimators": [150.0, 300.0, 500.0],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3.0, 5.0, 7.0],
    "num_leaves": [15.0, 31.0],
}


@dataclass
class CvResult:
    """Median MAPE for one hyperparameter combination, averaged across CV folds."""

    params: dict
    fold_median_mapes: list[float]
    mean_of_fold_medians: float


def _make_cv_fold(
    train_df: pd.DataFrame, cutoff: int, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into a CV-train/CV-validation pair at a given cutoff year."""
    cv_train = train_df[train_df["year"] <= cutoff]
    cv_val = train_df[(train_df["year"] > cutoff) & (train_df["year"] <= cutoff + horizon)]
    return cv_train, cv_val


def _score_params(params: dict, train_df: pd.DataFrame) -> CvResult:
    """Train and recursively evaluate one hyperparameter combination across all CV folds."""
    fold_medians = []

    for cutoff in CV_CUTOFFS:
        cv_train, cv_val = _make_cv_fold(train_df, cutoff, CV_HORIZON_YEARS)
        if cv_train.empty or cv_val.empty:
            continue

        x_train, y_train, w_train, _, y_val, val_meta, train_meta = prepare_gbm_features(
            cv_train, cv_val
        )
        if len(x_train) == 0 or len(val_meta) == 0:
            continue

        model = lgb.LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1, **params)
        model.fit(x_train, y_train, sample_weight=w_train)
        candidate = ModelCandidate(name="LightGBM-CV", model=model)

        forecast_years = sorted(val_meta["year"].unique().tolist())
        forecast_df = recursive_forecast(candidate, x_train, train_meta, forecast_years)

        actuals = val_meta.copy()
        actuals["ev_sales_share"] = y_val.to_numpy()
        merged = actuals.merge(forecast_df, on=["region_country", "mode", "year"], how="inner")

        if merged.empty:
            continue

        result = evaluate_predictions(merged)
        fold_medians.append(float(result.per_group_metrics["mape"].median()))

    mean_of_medians = sum(fold_medians) / len(fold_medians) if fold_medians else float("inf")
    return CvResult(
        params=params, fold_median_mapes=fold_medians, mean_of_fold_medians=mean_of_medians
    )


def run_grid_search(train_df: pd.DataFrame) -> list[CvResult]:
    """Evaluate every combination in PARAM_GRID via time-series CV, sorted best-first."""
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*PARAM_GRID.values()))

    int_params = {"n_estimators", "max_depth", "num_leaves"}
    results = []
    for combo in combinations:
        raw_params = dict(zip(keys, combo, strict=True))
        params = {k: int(v) if k in int_params else v for k, v in raw_params.items()}
        results.append(_score_params(params, train_df))

    return sorted(results, key=lambda r: r.mean_of_fold_medians)


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    train = pd.read_parquet(project_root / "data" / "processed" / "features_train.parquet")

    results = run_grid_search(train)

    print(f"Evaluated {len(results)} hyperparameter combinations via time-series CV\n")
    print("Top 5 combinations:")
    for r in results[:5]:
        print(
            f"  Mean-of-fold-medians MAPE: {r.mean_of_fold_medians:.2f}  |  {r.params}  |  folds: {r.fold_median_mapes}"
        )
