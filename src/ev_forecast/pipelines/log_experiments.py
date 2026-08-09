"""Re-run and log every model candidate to MLflow with consistent, unambiguous metrics.

Every run logs BOTH the aggregate metric (computed across all matched rows pooled
together) and the median-of-per-group metric (computed per country/mode, then take
the median across groups). These are genuinely different statistics -- aggregate is
dominated by whichever groups have the most rows, while median-of-per-group treats
every country equally. Logging both, clearly labeled, prevents ever comparing one
model's aggregate number to another model's per-group-median number by mistake --
exactly the kind of ambiguity that caused an incorrect conclusion earlier in this
project (see docs/modeling_notes.md).
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from ev_forecast.models.baseline import fit_all_curves, predict_with_curves
from ev_forecast.models.evaluation import EvaluationResult, evaluate_predictions
from ev_forecast.models.experiment_tracking import init_experiment, log_model_run
from ev_forecast.models.gbm_features import prepare_gbm_features
from ev_forecast.models.gbm_models import ModelCandidate, train_all_candidates
from ev_forecast.models.recursive_forecast import recursive_forecast

DEFAULT_GBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5}
TUNED_LIGHTGBM_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 7,
    "num_leaves": 15,
}
RANDOM_SEED = 42


def _metrics_from_result(result: EvaluationResult) -> dict[str, float]:
    """Build the standard, unambiguously-named metric set logged for every model."""
    return {
        "mape_aggregate": result.mape,
        "mape_median_per_group": float(result.per_group_metrics["mape"].median()),
        "rmse_aggregate": result.rmse,
        "rmse_median_per_group": float(result.per_group_metrics["rmse"].median()),
        "mae_aggregate": result.mae,
        "mae_median_per_group": float(result.per_group_metrics["mae"].median()),
    }


def log_baseline(train: pd.DataFrame, test: pd.DataFrame) -> str:
    fits = fit_all_curves(train)
    converged = sum(1 for f in fits.values() if f.converged)

    predictions = predict_with_curves(test, fits)
    predictions_clean = predictions.dropna(subset=["predicted_share"])
    result = evaluate_predictions(predictions_clean)

    return log_model_run(
        run_name="logistic-baseline",
        params={"model_type": "logistic_curve", "min_training_points": 4},
        metrics={
            **_metrics_from_result(result),
            "converged_groups": float(converged),
            "total_groups": float(len(fits)),
        },
        tags={"model_family": "baseline", "selected_for_production": "true"},
    )


def log_gbm_candidates(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    x_train, y_train, w_train, x_test, y_test, test_meta, train_meta = prepare_gbm_features(
        train, test
    )
    actuals = test_meta.copy()
    actuals["ev_sales_share"] = y_test.to_numpy()
    forecast_years = sorted(test_meta["year"].unique().tolist())

    run_ids = []
    candidates = train_all_candidates(x_train, y_train, w_train)
    for candidate in candidates:
        forecast_df = recursive_forecast(candidate, x_train, train_meta, forecast_years)
        merged = actuals.merge(forecast_df, on=["region_country", "mode", "year"], how="inner")
        result = evaluate_predictions(merged)

        run_name = f"{candidate.name.lower().replace(' ', '-')}-untuned"
        run_id = log_model_run(
            run_name=run_name,
            params={"model_type": candidate.name, **DEFAULT_GBM_PARAMS},
            metrics=_metrics_from_result(result),
            tags={"model_family": "gbm", "tuned": "false"},
        )
        run_ids.append(run_id)

    model = lgb.LGBMRegressor(
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_estimators=int(TUNED_LIGHTGBM_PARAMS["n_estimators"]),
        learning_rate=TUNED_LIGHTGBM_PARAMS["learning_rate"],
        max_depth=int(TUNED_LIGHTGBM_PARAMS["max_depth"]),
        num_leaves=int(TUNED_LIGHTGBM_PARAMS["num_leaves"]),
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    tuned_candidate = ModelCandidate(name="LightGBM-tuned", model=model)

    forecast_df = recursive_forecast(tuned_candidate, x_train, train_meta, forecast_years)
    merged = actuals.merge(forecast_df, on=["region_country", "mode", "year"], how="inner")
    result = evaluate_predictions(merged)

    run_id = log_model_run(
        run_name="lightgbm-tuned",
        params={"model_type": "LightGBM", **TUNED_LIGHTGBM_PARAMS},
        metrics=_metrics_from_result(result),
        tags={"model_family": "gbm", "tuned": "true"},
    )
    run_ids.append(run_id)

    return run_ids


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    init_experiment(project_root)

    train_df = pd.read_parquet(project_root / "data" / "processed" / "features_train.parquet")
    test_df = pd.read_parquet(project_root / "data" / "processed" / "features_test.parquet")

    baseline_run_id = log_baseline(train_df, test_df)
    print(f"Logged baseline: {baseline_run_id}")

    gbm_run_ids = log_gbm_candidates(train_df, test_df)
    print(f"Logged {len(gbm_run_ids)} GBM runs")
