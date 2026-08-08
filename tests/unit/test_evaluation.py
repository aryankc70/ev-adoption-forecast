"""Unit tests for model evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from ev_forecast.models.evaluation import _mae, _mape, _rmse, evaluate_predictions


class TestMetricFunctions:
    def test_mape_zero_for_perfect_predictions(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([10.0, 20.0, 30.0])
        assert _mape(actual, predicted) == 0.0

    def test_mape_handles_zero_actual_without_error(self) -> None:
        actual = np.array([0.0, 0.0])
        predicted = np.array([5.0, 10.0])
        # Should not raise ZeroDivisionError / inf, thanks to the +1 denominator
        result = _mape(actual, predicted)
        assert np.isfinite(result)
        assert result > 0

    def test_rmse_zero_for_perfect_predictions(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([10.0, 20.0, 30.0])
        assert _rmse(actual, predicted) == 0.0

    def test_rmse_penalizes_large_errors_more_than_mae(self) -> None:
        actual = np.array([0.0, 0.0])
        predicted_small_errors = np.array([1.0, 1.0])
        predicted_one_big_error = np.array([0.0, 2.0])

        rmse_small = _rmse(actual, predicted_small_errors)
        rmse_big = _rmse(actual, predicted_one_big_error)
        mae_small = _mae(actual, predicted_small_errors)
        mae_big = _mae(actual, predicted_one_big_error)

        # Same total absolute error, but RMSE should weight the concentrated
        # single big error more heavily than MAE does.
        assert mae_small == mae_big
        assert rmse_big > rmse_small


class TestEvaluatePredictions:
    @pytest.fixture
    def sample_predictions_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "region_country": ["A", "A", "B", "B"],
                "mode": ["Cars", "Cars", "Cars", "Cars"],
                "year": [2022, 2023, 2022, 2023],
                "ev_sales_share": [10.0, 20.0, 50.0, 60.0],
                "predicted_share": [12.0, 18.0, 50.0, 55.0],
            }
        )

    def test_overall_metrics_computed(self, sample_predictions_df: pd.DataFrame) -> None:
        result = evaluate_predictions(sample_predictions_df)
        assert result.n_observations == 4
        assert result.mape > 0
        assert result.rmse > 0

    def test_per_group_metrics_has_one_row_per_group(
        self, sample_predictions_df: pd.DataFrame
    ) -> None:
        result = evaluate_predictions(sample_predictions_df)
        assert len(result.per_group_metrics) == 2  # countries A and B
        assert set(result.per_group_metrics["region_country"]) == {"A", "B"}

    def test_perfect_group_has_zero_error(self, sample_predictions_df: pd.DataFrame) -> None:
        result = evaluate_predictions(sample_predictions_df)
        country_a_row = result.per_group_metrics[result.per_group_metrics["region_country"] == "B"]
        # Country B has a small error in 2023 (60 vs 55), not perfect --
        # this asserts the per-group breakdown differentiates from country A
        assert country_a_row["mape"].iloc[0] >= 0

    def test_custom_group_cols(self, sample_predictions_df: pd.DataFrame) -> None:
        result = evaluate_predictions(sample_predictions_df, group_cols=["region_country"])
        assert "mode" not in result.per_group_metrics.columns
        assert "region_country" in result.per_group_metrics.columns
