"""Unit tests for recursive multi-step forecasting."""

import numpy as np
import pandas as pd

from ev_forecast.models.gbm_features import prepare_gbm_features
from ev_forecast.models.gbm_models import ModelCandidate
from ev_forecast.models.recursive_forecast import recursive_forecast


class _StubModel:
    """A fake model whose predict() we control exactly, to test the forecasting loop in isolation."""

    def __init__(self, fixed_prediction: float) -> None:
        self.fixed_prediction = fixed_prediction
        self.received_inputs: list[pd.DataFrame] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.received_inputs.append(X.copy())
        return np.array([self.fixed_prediction])


def _synthetic_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "region_country": ["A", "A", "A"],
            "mode": ["Cars", "Cars", "Cars"],
            "year": [2019, 2020, 2021],
            "ev_sales_share": [1.0, 2.0, 3.0],
            "ev_sales_share_lag1": [None, 1.0, 2.0],
            "ev_sales_share_yoy_change": [None, 1.0, 1.0],
            "takeoff_year": [2019.0, 2019.0, 2019.0],
            "years_since_takeoff": [0.0, 1.0, 2.0],
            "cumulative_years_tracked": [1, 2, 3],
            "sample_weight": [1.0, 1.0, 1.0],
            "ev_sales_volume": [100.0, 200.0, 300.0],
        }
    )
    test = pd.DataFrame(
        {
            "region_country": ["A", "A"],
            "mode": ["Cars", "Cars"],
            "year": [2022, 2023],
            "ev_sales_share": [4.0, 5.0],
            "ev_sales_share_lag1": [3.0, 4.0],
            "ev_sales_share_yoy_change": [1.0, 1.0],
            "takeoff_year": [2019.0, 2019.0],
            "years_since_takeoff": [3.0, 4.0],
            "cumulative_years_tracked": [4, 5],
            "sample_weight": [1.0, 1.0],
            "ev_sales_volume": [400.0, 500.0],
        }
    )
    return train, test


class TestRecursiveForecast:
    def test_returns_one_row_per_group_per_forecast_year(self) -> None:
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, train_meta = prepare_gbm_features(train, test)

        stub = ModelCandidate(name="Stub", model=_StubModel(fixed_prediction=5.0))
        result = recursive_forecast(stub, x_train, train_meta, forecast_years=[2022, 2023])

        assert len(result) == 2
        assert set(result["year"]) == {2022, 2023}
        assert (result["region_country"] == "A").all()

    def test_second_step_lag_uses_first_steps_prediction_not_ground_truth(self) -> None:
        """
        This is the core leakage-prevention test: year 2023's lag1 input must be
        the model's own 2022 prediction (5.0, from the stub), never the real
        actual 2022 value (which the model must not have access to).
        """
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, train_meta = prepare_gbm_features(train, test)

        stub_model = _StubModel(fixed_prediction=5.0)
        stub = ModelCandidate(name="Stub", model=stub_model)
        recursive_forecast(stub, x_train, train_meta, forecast_years=[2022, 2023])

        # Two predict() calls: one per forecast year
        assert len(stub_model.received_inputs) == 2
        second_call_input = stub_model.received_inputs[1]
        # The lag1 feature fed into the second call must equal the FIRST call's
        # prediction (5.0, since our stub always returns 5.0), not any real ground truth.
        assert second_call_input["ev_sales_share_lag1"].iloc[0] == 5.0

    def test_cumulative_years_tracked_increments_each_step(self) -> None:
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, train_meta = prepare_gbm_features(train, test)

        stub_model = _StubModel(fixed_prediction=5.0)
        stub = ModelCandidate(name="Stub", model=stub_model)
        recursive_forecast(stub, x_train, train_meta, forecast_years=[2022, 2023])

        first_call_years_tracked = stub_model.received_inputs[0]["cumulative_years_tracked"].iloc[0]
        second_call_years_tracked = stub_model.received_inputs[1]["cumulative_years_tracked"].iloc[
            0
        ]
        assert second_call_years_tracked == first_call_years_tracked + 1

    def test_no_nan_passed_to_model(self) -> None:
        """Every model in our comparison must receive fully imputed, finite inputs."""
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, train_meta = prepare_gbm_features(train, test)

        stub_model = _StubModel(fixed_prediction=5.0)
        stub = ModelCandidate(name="Stub", model=stub_model)
        recursive_forecast(stub, x_train, train_meta, forecast_years=[2022, 2023])

        for call_input in stub_model.received_inputs:
            assert not call_input.isna().any().any()
