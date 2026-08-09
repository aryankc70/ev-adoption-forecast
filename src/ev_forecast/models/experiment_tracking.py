"""MLflow experiment tracking helpers, shared across all model types."""

from pathlib import Path
from typing import Any

import mlflow

EXPERIMENT_NAME = "ev-adoption-forecast"


def get_tracking_uri(project_root: Path) -> str:
    """
    Local SQLite-backed tracking store under the project root.

    SQLite is used rather than a remote server since this project runs entirely
    locally -- it still gives full run history, metric comparison, and querying.
    MLflow's plain filesystem store is in maintenance mode as of MLflow 3.x, so
    SQLite is the current recommended local backend, not just a stylistic choice.
    """
    db_path = project_root / "mlflow.db"
    return f"sqlite:///{db_path}"


def init_experiment(project_root: Path) -> None:
    """Point MLflow at the local tracking store and select/create the experiment."""
    mlflow.set_tracking_uri(get_tracking_uri(project_root))
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_model_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log one model run's parameters and metrics under a consistent schema, so
    every model type (baseline, LightGBM, XGBoost, sklearn GBM, tuned variants)
    is comparable in the same experiment regardless of its underlying library.

    Returns the MLflow run ID, so callers can reference this run later
    (e.g. to mark the winning run, or load it in Phase 8's serving API).
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        return str(run.info.run_id)
