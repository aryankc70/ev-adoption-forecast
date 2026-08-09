"""Recursive multi-step forecasting for GBM models.

Predicting year N+2 from a fixed train cutoff must not use the real actual
value from year N+1 as a lag feature -- that value would not exist yet at
real forecast time. Instead, each step's prediction becomes the next step's
lag input, exactly matching how the model would be used in production.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ev_forecast.models.gbm_models import ModelCandidate

TAKEOFF_THRESHOLD_PCT = 1.0


@dataclass
class GroupState:
    """Mutable forecasting state for one (country, mode) group between steps."""

    region_country: str
    mode: str
    last_share: float
    prior_share: float | None
    takeoff_year: float | None
    cumulative_years_tracked: int
    template_row: pd.Series  # one-hot + all feature columns from the last known training row


def _build_group_states(x_train: pd.DataFrame, train_meta: pd.DataFrame) -> list[GroupState]:
    """Extract each group's most recent known state at the train/test boundary."""
    states = []
    # x_train holds the one-hot encoded feature columns (source of truth for the
    # model's inputs); train_meta holds the identifying/raw columns needed to pick
    # each group's latest row. Both share year/takeoff_year/cumulative_years_tracked,
    # so keep only train_meta's copies of those to avoid an ambiguous merge.
    x_train_unique_cols = x_train.drop(columns=["year", "takeoff_year", "cumulative_years_tracked"])
    combined = pd.concat([train_meta.reset_index(drop=True), x_train_unique_cols], axis=1)

    for (country, mode), group in combined.groupby(["region_country", "mode"]):
        group_sorted = group.sort_values("year")
        last_row = group_sorted.iloc[-1]
        prior_share = (
            float(group_sorted.iloc[-2]["ev_sales_share"]) if len(group_sorted) >= 2 else None
        )

        states.append(
            GroupState(
                region_country=str(country),
                mode=str(mode),
                last_share=float(last_row["ev_sales_share"]),
                prior_share=prior_share,
                takeoff_year=(
                    float(last_row["takeoff_year"]) if pd.notna(last_row["takeoff_year"]) else None
                ),
                cumulative_years_tracked=int(last_row["cumulative_years_tracked"]),
                template_row=last_row.drop(
                    labels=[
                        "region_country",
                        "mode",
                        "year",
                        "ev_sales_share",
                        "takeoff_year",
                        "cumulative_years_tracked",
                    ]
                ),
            )
        )
    return states


def recursive_forecast(
    candidate: ModelCandidate,
    x_train: pd.DataFrame,
    train_meta: pd.DataFrame,
    forecast_years: list[int],
) -> pd.DataFrame:
    """
    Forecast forward year-by-year, using each step's own prediction as the
    next step's lag input. Returns one row per (country, mode, forecast_year).
    """
    states = _build_group_states(x_train, train_meta)
    forecast_years_sorted = sorted(forecast_years)
    rows = []

    for state in states:
        current_last_share = state.last_share
        current_prior_share = state.prior_share
        current_years_tracked = state.cumulative_years_tracked

        for year in forecast_years_sorted:
            current_years_tracked += 1
            lag1 = current_last_share
            yoy_change = (
                current_last_share - current_prior_share
                if current_prior_share is not None
                else np.nan
            )
            years_since_takeoff = (
                year - state.takeoff_year if state.takeoff_year is not None else np.nan
            )

            feature_row = state.template_row.copy()
            feature_row["year"] = float(year)
            feature_row["ev_sales_share_lag1"] = lag1
            feature_row["ev_sales_share_yoy_change"] = yoy_change
            feature_row["takeoff_year"] = (
                state.takeoff_year if state.takeoff_year is not None else np.nan
            )
            feature_row["years_since_takeoff"] = years_since_takeoff
            feature_row["cumulative_years_tracked"] = current_years_tracked

            x_row = pd.DataFrame([feature_row])[x_train.columns]
            # Impute any remaining NaN (e.g. years_since_takeoff for a country
            # that hasn't taken off yet) with 0 consistently across all three
            # candidate models. LightGBM/XGBoost can handle NaN natively, but
            # sklearn's GradientBoostingRegressor cannot -- imputing uniformly
            # for all three keeps the comparison fair rather than letting some
            # models see different effective inputs than others.
            x_row = x_row.fillna(0.0)
            predicted_share = float(candidate.model.predict(x_row)[0])

            rows.append(
                {
                    "region_country": state.region_country,
                    "mode": state.mode,
                    "year": year,
                    "predicted_share": predicted_share,
                }
            )

            current_prior_share = current_last_share
            current_last_share = predicted_share

    return pd.DataFrame(rows)
