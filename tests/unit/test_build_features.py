"""Unit tests for feature engineering transformations."""

import pandas as pd

from ev_forecast.features.build_features import (
    _add_lag_features,
    _add_sample_weight,
    _add_takeoff_features,
    _pivot_target,
    train_test_split_by_year,
)


def _synthetic_clean_df() -> pd.DataFrame:
    """A small long-format dataset mimicking the Phase 2 cleaned output."""
    return pd.DataFrame(
        [
            {
                "region_country": "Testland",
                "mode": "Cars",
                "parameter": "EV sales share",
                "powertrain": "EV",
                "year": 2018,
                "value": 0.5,
            },
            {
                "region_country": "Testland",
                "mode": "Cars",
                "parameter": "EV sales share",
                "powertrain": "EV",
                "year": 2019,
                "value": 1.5,
            },
            {
                "region_country": "Testland",
                "mode": "Cars",
                "parameter": "EV sales share",
                "powertrain": "EV",
                "year": 2020,
                "value": 3.0,
            },
            {
                "region_country": "Testland",
                "mode": "Cars",
                "parameter": "EV sales",
                "powertrain": "EV",
                "year": 2018,
                "value": 500,
            },
            {
                "region_country": "Testland",
                "mode": "Cars",
                "parameter": "EV sales",
                "powertrain": "EV",
                "year": 2019,
                "value": 1500,
            },
            {
                "region_country": "Testland",
                "mode": "Cars",
                "parameter": "EV sales",
                "powertrain": "EV",
                "year": 2020,
                "value": 3000,
            },
            # A different country, to test grouping doesn't leak across countries
            {
                "region_country": "OtherLand",
                "mode": "Cars",
                "parameter": "EV sales share",
                "powertrain": "EV",
                "year": 2018,
                "value": 10.0,
            },
            {
                "region_country": "OtherLand",
                "mode": "Cars",
                "parameter": "EV sales share",
                "powertrain": "EV",
                "year": 2019,
                "value": 20.0,
            },
        ]
    )


class TestPivotTarget:
    def test_reshapes_to_one_row_per_country_mode_year(self) -> None:
        result = _pivot_target(_synthetic_clean_df())
        assert len(result) == 5  # 3 Testland + 2 OtherLand
        assert set(result.columns) >= {
            "region_country",
            "mode",
            "year",
            "ev_sales_share",
            "ev_sales_volume",
        }

    def test_share_and_volume_align_on_correct_year(self) -> None:
        result = _pivot_target(_synthetic_clean_df())
        row_2019 = result[(result["region_country"] == "Testland") & (result["year"] == 2019)]
        assert row_2019.iloc[0]["ev_sales_share"] == 1.5
        assert row_2019.iloc[0]["ev_sales_volume"] == 1500


class TestLagFeatures:
    def test_lag1_is_prior_year_value(self) -> None:
        pivoted = _pivot_target(_synthetic_clean_df())
        result = _add_lag_features(pivoted)
        row_2020 = result[(result["region_country"] == "Testland") & (result["year"] == 2020)]
        assert row_2020.iloc[0]["ev_sales_share_lag1"] == 1.5
        assert row_2020.iloc[0]["ev_sales_share_yoy_change"] == 1.5  # 3.0 - 1.5

    def test_first_year_has_no_lag(self) -> None:
        pivoted = _pivot_target(_synthetic_clean_df())
        result = _add_lag_features(pivoted)
        row_2018 = result[(result["region_country"] == "Testland") & (result["year"] == 2018)]
        assert pd.isna(row_2018.iloc[0]["ev_sales_share_lag1"])

    def test_lag_does_not_leak_across_countries(self) -> None:
        pivoted = _pivot_target(_synthetic_clean_df())
        result = _add_lag_features(pivoted)
        # OtherLand's first year (2018) must NOT pick up Testland's last value as a lag
        other_2018 = result[(result["region_country"] == "OtherLand") & (result["year"] == 2018)]
        assert pd.isna(other_2018.iloc[0]["ev_sales_share_lag1"])


class TestTakeoffFeatures:
    def test_takeoff_year_is_first_year_above_threshold(self) -> None:
        pivoted = _pivot_target(_synthetic_clean_df())
        lagged = _add_lag_features(pivoted)
        result = _add_takeoff_features(lagged)
        testland = result[result["region_country"] == "Testland"]
        # Testland crosses 1.0% threshold in 2019 (0.5 -> 1.5)
        assert (testland["takeoff_year"] == 2019).all()

    def test_years_since_takeoff_is_relative_to_takeoff_year(self) -> None:
        pivoted = _pivot_target(_synthetic_clean_df())
        lagged = _add_lag_features(pivoted)
        result = _add_takeoff_features(lagged)
        row_2020 = result[(result["region_country"] == "Testland") & (result["year"] == 2020)]
        assert row_2020.iloc[0]["years_since_takeoff"] == 1  # 2020 - 2019

    def test_cumulative_years_tracked_increments_per_country(self) -> None:
        pivoted = _pivot_target(_synthetic_clean_df())
        lagged = _add_lag_features(pivoted)
        result = _add_takeoff_features(lagged)
        testland = result.sort_values("year")
        testland = testland[testland["region_country"] == "Testland"]
        assert testland["cumulative_years_tracked"].tolist() == [1, 2, 3]


class TestSampleWeight:
    def test_larger_volume_gets_larger_weight(self) -> None:
        df = pd.DataFrame(
            {
                "region_country": ["Small", "Large"],
                "mode": ["Cars", "Cars"],
                "year": [2020, 2020],
                "ev_sales_volume": [100.0, 1_000_000.0],
            }
        )
        result = _add_sample_weight(df)
        small_weight = result[result["region_country"] == "Small"]["sample_weight"].iloc[0]
        large_weight = result[result["region_country"] == "Large"]["sample_weight"].iloc[0]
        assert large_weight > small_weight

    def test_weight_is_log_compressed_not_linear(self) -> None:
        df = pd.DataFrame(
            {
                "region_country": ["Small", "Large"],
                "mode": ["Cars", "Cars"],
                "year": [2020, 2020],
                "ev_sales_volume": [100.0, 1_000_000.0],
            }
        )
        result = _add_sample_weight(df)
        small_weight = result[result["region_country"] == "Small"]["sample_weight"].iloc[0]
        large_weight = result[result["region_country"] == "Large"]["sample_weight"].iloc[0]

        raw_ratio = 1_000_000.0 / 100.0
        weight_ratio = large_weight / small_weight
        assert weight_ratio < raw_ratio / 10  # log-scaling should massively compress the ratio

    def test_missing_volume_does_not_crash(self) -> None:
        df = pd.DataFrame(
            {
                "region_country": ["NoVolume"],
                "mode": ["Cars"],
                "year": [2020],
                "ev_sales_volume": [None],
            }
        )
        result = _add_sample_weight(df)
        assert result["sample_weight"].iloc[0] == 0.0


class TestTrainTestSplit:
    def test_split_respects_year_boundary(self) -> None:
        df = pd.DataFrame(
            {
                "region_country": ["A", "A", "A", "A"],
                "mode": ["Cars"] * 4,
                "year": [2020, 2021, 2022, 2023],
                "ev_sales_share": [1.0, 2.0, 3.0, 4.0],
            }
        )
        train, test = train_test_split_by_year(df, split_year=2021)
        assert set(train["year"]) == {2020, 2021}
        assert set(test["year"]) == {2022, 2023}

    def test_split_applies_uniformly_across_countries(self) -> None:
        df = pd.DataFrame(
            {
                "region_country": ["A", "A", "B", "B"],
                "mode": ["Cars"] * 4,
                "year": [2020, 2022, 2020, 2022],
                "ev_sales_share": [1.0, 2.0, 3.0, 4.0],
            }
        )
        train, test = train_test_split_by_year(df, split_year=2021)
        # Both countries should be split at the same year, not e.g. by row count
        assert set(train["region_country"]) == {"A", "B"}
        assert set(test["region_country"]) == {"A", "B"}

    def test_no_row_appears_in_both_sets(self) -> None:
        df = pd.DataFrame(
            {
                "region_country": ["A"] * 6,
                "mode": ["Cars"] * 6,
                "year": [2018, 2019, 2020, 2021, 2022, 2023],
                "ev_sales_share": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        train, test = train_test_split_by_year(df, split_year=2021)
        overlap = set(train["year"]) & set(test["year"])
        assert overlap == set()
