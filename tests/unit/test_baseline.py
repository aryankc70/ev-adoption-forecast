"""Unit tests for the logistic diffusion baseline model."""

import numpy as np
import pandas as pd
from pytest import approx as pytest_approx

from ev_forecast.models.baseline import (
    fit_all_curves,
    fit_logistic_curve,
    predict_with_curves,
)


def _synthetic_logistic_data(
    saturation: float = 90.0, growth_rate: float = 0.8, inflection_year: float = 2018.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean, noise-free points from a known logistic curve, for testing recovery."""
    years = np.arange(2010, 2022, dtype=float)
    shares = saturation / (1 + np.exp(-growth_rate * (years - inflection_year)))
    return years, shares


class TestFitLogisticCurve:
    def test_recovers_known_parameters_from_clean_data(self) -> None:
        true_saturation, true_growth_rate, true_inflection = 90.0, 0.8, 2018.0
        years, shares = _synthetic_logistic_data(true_saturation, true_growth_rate, true_inflection)

        result = fit_logistic_curve(years, shares)

        assert result.converged is True
        assert result.saturation is not None
        assert result.saturation == pytest_approx(true_saturation, abs=1.0)

    def test_fails_gracefully_with_too_few_points(self) -> None:
        years = np.array([2020.0, 2021.0])
        shares = np.array([1.0, 2.0])

        result = fit_logistic_curve(years, shares)

        assert result.converged is False
        assert result.failure_reason is not None
        assert "fewer than" in result.failure_reason

    def test_does_not_raise_on_degenerate_flat_data(self) -> None:
        # All values identical -- no curvature at all, a real edge case in our data
        years = np.array([2018.0, 2019.0, 2020.0, 2021.0])
        shares = np.array([0.0, 0.0, 0.0, 0.0])

        # Should not raise, regardless of whether it converges
        result = fit_logistic_curve(years, shares)
        assert result is not None


class TestFitAllCurves:
    def test_fits_one_curve_per_country_mode_group(self) -> None:
        years, shares = _synthetic_logistic_data()
        df = pd.DataFrame(
            {
                "region_country": ["A"] * len(years) + ["B"] * len(years),
                "mode": ["Cars"] * (len(years) * 2),
                "year": np.concatenate([years, years]),
                "ev_sales_share": np.concatenate([shares, shares]),
            }
        )

        fits = fit_all_curves(df)

        assert set(fits.keys()) == {("A", "Cars"), ("B", "Cars")}
        assert fits[("A", "Cars")].converged is True
        assert fits[("A", "Cars")].region_country == "A"


class TestPredictWithCurves:
    def test_predicts_using_fitted_curve(self) -> None:
        years, shares = _synthetic_logistic_data()
        train_df = pd.DataFrame(
            {
                "region_country": ["A"] * len(years),
                "mode": ["Cars"] * len(years),
                "year": years,
                "ev_sales_share": shares,
            }
        )
        fits = fit_all_curves(train_df)

        test_df = pd.DataFrame(
            {
                "region_country": ["A"],
                "mode": ["Cars"],
                "year": [2022.0],
                "ev_sales_share": [91.0],
            }
        )
        result = predict_with_curves(test_df, fits)

        assert "predicted_share" in result.columns
        assert not pd.isna(result["predicted_share"].iloc[0])

    def test_unknown_group_gets_nan_prediction(self) -> None:
        test_df = pd.DataFrame(
            {
                "region_country": ["NeverSeen"],
                "mode": ["Cars"],
                "year": [2022.0],
                "ev_sales_share": [5.0],
            }
        )
        result = predict_with_curves(test_df, fits={})

        assert pd.isna(result["predicted_share"].iloc[0])

    def test_non_converged_fit_gets_nan_prediction(self) -> None:
        from ev_forecast.models.baseline import CurveFitResult

        test_df = pd.DataFrame(
            {
                "region_country": ["A"],
                "mode": ["Cars"],
                "year": [2022.0],
                "ev_sales_share": [5.0],
            }
        )
        failed_fit = CurveFitResult(
            region_country="A", mode="Cars", converged=False, failure_reason="test failure"
        )
        result = predict_with_curves(test_df, fits={("A", "Cars"): failed_fit})

        assert pd.isna(result["predicted_share"].iloc[0])
