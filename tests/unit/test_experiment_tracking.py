"""Unit tests for MLflow experiment tracking helpers."""

from pathlib import Path

import mlflow

from ev_forecast.models.experiment_tracking import (
    EXPERIMENT_NAME,
    get_tracking_uri,
    init_experiment,
    log_model_run,
)


class TestGetTrackingUri:
    def test_returns_sqlite_uri(self, tmp_path: Path) -> None:
        uri = get_tracking_uri(tmp_path)
        assert uri.startswith("sqlite:///")
        assert "mlflow.db" in uri


class TestLogModelRun:
    def test_logs_run_and_returns_run_id(self, tmp_path: Path) -> None:
        init_experiment(tmp_path)

        run_id = log_model_run(
            run_name="test-run",
            params={"model_type": "test_model"},
            metrics={"rmse_aggregate": 1.5, "mape_aggregate": 10.0},
        )

        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_logged_metrics_are_retrievable(self, tmp_path: Path) -> None:
        init_experiment(tmp_path)

        run_id = log_model_run(
            run_name="test-run-2",
            params={"model_type": "test_model"},
            metrics={"rmse_aggregate": 2.5, "mape_aggregate": 20.0},
        )

        run = mlflow.get_run(run_id)
        assert run.data.metrics["rmse_aggregate"] == 2.5
        assert run.data.params["model_type"] == "test_model"

    def test_tags_are_optional(self, tmp_path: Path) -> None:
        init_experiment(tmp_path)

        run_id = log_model_run(
            run_name="test-run-no-tags",
            params={"model_type": "test_model"},
            metrics={"rmse_aggregate": 1.0},
        )
        # Should not raise, even with no tags argument passed
        run = mlflow.get_run(run_id)
        assert run.info.run_id == run_id

    def test_experiment_name_is_set(self, tmp_path: Path) -> None:
        init_experiment(tmp_path)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        assert experiment is not None
        assert experiment.name == EXPERIMENT_NAME
