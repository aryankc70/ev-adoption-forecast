"""Unit tests for GBM feature preparation."""

import pandas as pd

from ev_forecast.models.gbm_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_gbm_features,
)


def _synthetic_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "region_country": ["A", "A", "B", "B"],
            "mode": ["Cars", "Cars", "Cars", "Cars"],
            "year": [2019, 2020, 2019, 2020],
            "ev_sales_share": [1.0, 2.0, 5.0, 6.0],
            "ev_sales_share_lag1": [None, 1.0, None, 5.0],
            "ev_sales_share_yoy_change": [None, 1.0, None, 1.0],
            "takeoff_year": [2020.0, 2020.0, 2019.0, 2019.0],
            "years_since_takeoff": [-1.0, 0.0, 0.0, 1.0],
            "cumulative_years_tracked": [1, 2, 1, 2],
            "sample_weight": [1.0, 1.5, 2.0, 2.5],
            "ev_sales_volume": [100.0, 150.0, 200.0, 250.0],
        }
    )
    test = pd.DataFrame(
        {
            "region_country": ["A", "B"],
            "mode": ["Cars", "Cars"],
            "year": [2021, 2021],
            "ev_sales_share": [3.0, 7.0],
            "ev_sales_share_lag1": [2.0, 6.0],
            "ev_sales_share_yoy_change": [1.0, 1.0],
            "takeoff_year": [2020.0, 2019.0],
            "years_since_takeoff": [1.0, 2.0],
            "cumulative_years_tracked": [3, 3],
            "sample_weight": [2.0, 3.0],
            "ev_sales_volume": [200.0, 300.0],
        }
    )
    return train, test


class TestPrepareGbmFeatures:
    def test_drops_rows_missing_lag_features(self) -> None:
        train, test = _synthetic_train_test()
        x_train, y_train, w_train, x_test, y_test, test_meta, train_meta = prepare_gbm_features(
            train, test
        )
        # Each group's first year (no lag1 available) should be dropped
        assert len(x_train) == 2  # only the 2020 rows survive
        assert len(x_test) == 2  # both 2021 rows have valid lag1

    def test_train_and_test_have_identical_columns(self) -> None:
        train, test = _synthetic_train_test()
        x_train, _, _, x_test, _, _, _ = prepare_gbm_features(train, test)
        assert list(x_train.columns) == list(x_test.columns)

    def test_categorical_columns_are_one_hot_encoded(self) -> None:
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, _ = prepare_gbm_features(train, test)
        assert any(c.startswith("region_country_") for c in x_train.columns)
        assert any(c.startswith("mode_") for c in x_train.columns)
        for raw_col in CATEGORICAL_FEATURES:
            assert raw_col not in x_train.columns

    def test_target_and_weight_excluded_from_features(self) -> None:
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, _ = prepare_gbm_features(train, test)
        assert "ev_sales_share" not in x_train.columns
        assert "sample_weight" not in x_train.columns
        assert "ev_sales_volume" not in x_train.columns

    def test_numeric_features_present(self) -> None:
        train, test = _synthetic_train_test()
        x_train, _, _, _, _, _, _ = prepare_gbm_features(train, test)
        for col in NUMERIC_FEATURES:
            assert col in x_train.columns

    def test_test_meta_preserves_identifiers(self) -> None:
        train, test = _synthetic_train_test()
        _, _, _, _, _, test_meta, _ = prepare_gbm_features(train, test)
        assert set(test_meta.columns) >= {"region_country", "mode", "year"}
        assert len(test_meta) == 2
