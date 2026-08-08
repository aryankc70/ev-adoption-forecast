"""Bass/logistic diffusion baseline model, fit per (country, mode)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DEFAULT_SATURATION = 100.0
MIN_TRAINING_POINTS = 4


@dataclass
class CurveFitResult:
    """Fitted logistic curve parameters for one (country, mode) group."""

    region_country: str
    mode: str
    converged: bool
    saturation: float | None = None
    growth_rate: float | None = None
    inflection_year: float | None = None
    failure_reason: str | None = None


def _logistic(
    t: np.ndarray, saturation: float, growth_rate: float, inflection_year: float
) -> np.ndarray:
    return saturation / (1 + np.exp(-growth_rate * (t - inflection_year)))


def fit_logistic_curve(years: np.ndarray, shares: np.ndarray) -> CurveFitResult:
    """
    Fit a logistic S-curve to one country/mode's historical share data.

    Returns None-equivalent (converged=False) rather than raising when the
    fit fails -- this is expected and common for countries with too little
    data or no visible curvature yet, and the pipeline must continue for
    all other countries rather than crashing.
    """
    if len(years) < MIN_TRAINING_POINTS:
        return CurveFitResult(
            region_country="",
            mode="",
            converged=False,
            failure_reason=f"fewer than {MIN_TRAINING_POINTS} data points",
        )

    try:
        # Initial guesses: saturation near 100%, moderate growth rate,
        # inflection at the midpoint of the observed year range.
        p0 = [DEFAULT_SATURATION, 0.5, float(np.median(years))]
        bounds = ([1.0, 0.01, years.min() - 20], [100.0, 5.0, years.max() + 30])

        params, _ = curve_fit(_logistic, years, shares, p0=p0, bounds=bounds, maxfev=5000)
        saturation, growth_rate, inflection_year = params

        return CurveFitResult(
            region_country="",
            mode="",
            converged=True,
            saturation=float(saturation),
            growth_rate=float(growth_rate),
            inflection_year=float(inflection_year),
        )
    except (RuntimeError, ValueError) as e:
        return CurveFitResult(
            region_country="",
            mode="",
            converged=False,
            failure_reason=str(e),
        )


def fit_all_curves(train_df: pd.DataFrame) -> dict[tuple[str, str], CurveFitResult]:
    """Fit a logistic curve independently for every (region_country, mode) group."""
    results: dict[tuple[str, str], CurveFitResult] = {}

    for group_key, group in train_df.groupby(["region_country", "mode"]):
        country, mode = str(group_key[0]), str(group_key[1])
        group_sorted = group.sort_values("year")
        years = group_sorted["year"].to_numpy(dtype=float)
        shares = group_sorted["ev_sales_share"].to_numpy(dtype=float)

        result = fit_logistic_curve(years, shares)
        result.region_country = country
        result.mode = mode
        results[(country, mode)] = result

    return results


def predict_with_curves(
    test_df: pd.DataFrame, fits: dict[tuple[str, str], CurveFitResult]
) -> pd.DataFrame:
    """
    Generate predictions for the test set using previously fitted curves.

    Rows for (country, mode) groups whose fit failed to converge get NaN
    predictions rather than being silently dropped, so downstream evaluation
    can see exactly which groups the baseline couldn't cover.
    """
    df = test_df.copy()
    predictions = []

    for _, row in df.iterrows():
        key = (row["region_country"], row["mode"])
        fit = fits.get(key)

        if fit is None or not fit.converged:
            predictions.append(np.nan)
            continue

        # converged=True guarantees these are populated (see fit_logistic_curve),
        # but mypy can't infer that from the dataclass alone -- assert makes it explicit.
        assert fit.saturation is not None
        assert fit.growth_rate is not None
        assert fit.inflection_year is not None

        pred = _logistic(
            np.array([row["year"]], dtype=float),
            fit.saturation,
            fit.growth_rate,
            fit.inflection_year,
        )[0]
        predictions.append(pred)

    df["predicted_share"] = predictions
    return df
