"""Build model-ready features from cleaned EV adoption data."""

from pathlib import Path

import numpy as np
import pandas as pd

TAKEOFF_THRESHOLD_PCT = 1.0


def _pivot_target(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape long-format share data into one row per (country, mode, year)."""
    share = df[(df["parameter"] == "EV sales share") & (df["powertrain"] == "EV")][
        ["region_country", "mode", "year", "value"]
    ].rename(columns={"value": "ev_sales_share"})

    volume = df[(df["parameter"] == "EV sales") & (df["powertrain"] == "EV")][
        ["region_country", "mode", "year", "value"]
    ].rename(columns={"value": "ev_sales_volume"})

    merged = share.merge(volume, on=["region_country", "mode", "year"], how="left")
    return merged


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and year-over-year change features. Strictly backward-looking."""
    df = df.sort_values(["region_country", "mode", "year"]).copy()
    group_cols = ["region_country", "mode"]

    df["ev_sales_share_lag1"] = df.groupby(group_cols)["ev_sales_share"].shift(1)
    df["ev_sales_share_yoy_change"] = df["ev_sales_share"] - df["ev_sales_share_lag1"]

    return df


def _add_takeoff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add years-since-takeoff and data-maturity features."""
    df = df.sort_values(["region_country", "mode", "year"]).copy()
    group_cols = ["region_country", "mode"]

    # Year is NaN wherever share hasn't reached the takeoff threshold yet;
    # taking the group-wise min gives the first year each group crossed it.
    year_if_above_threshold = df["year"].where(df["ev_sales_share"] >= TAKEOFF_THRESHOLD_PCT)
    df["takeoff_year"] = df.groupby(group_cols)["year"].transform(
        lambda _: year_if_above_threshold.loc[_.index].min()
    )
    df["years_since_takeoff"] = df["year"] - df["takeoff_year"]

    df["cumulative_years_tracked"] = df.groupby(group_cols).cumcount() + 1

    return df


def _add_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a log-scaled sample weight based on market size (EV sales volume).

    Raw volume is heavily right-skewed (large markets can be 1000x+ bigger
    than small ones), so weighting directly by volume would let a handful of
    large countries dominate model loss entirely. Log-scaling compresses that
    range while still giving bigger, more reliable markets proportionally more
    influence than small, noisy ones like Iceland (see docs/eda_findings.md).
    """
    df = df.copy()
    df["sample_weight"] = np.log1p(df["ev_sales_volume"].fillna(0).astype(float))
    return df


TRAIN_TEST_SPLIT_YEAR = 2021


def train_test_split_by_year(
    df: pd.DataFrame, split_year: int = TRAIN_TEST_SPLIT_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature table into train/test sets by year, never by random row.

    Time series data must never be split randomly -- a random split would let
    the model train on 2023 data for one country while testing on 2020 data for
    another, which leaks future information and produces an unrealistically
    optimistic evaluation. Splitting strictly by year, applied uniformly across
    all countries, matches how the model will actually be used in production:
    trained on the past, evaluated on the future.
    """
    train = df[df["year"] <= split_year].copy()
    test = df[df["year"] > split_year].copy()
    return train, test


def build_feature_table(
    clean_data_path: Path, output_path: Path, train_output_path: Path, test_output_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature engineering pipeline: clean parquet -> model-ready train/test tables."""
    df = pd.read_parquet(clean_data_path)

    features = _pivot_target(df)
    features = _add_lag_features(features)
    features = _add_takeoff_features(features)
    features = _add_sample_weight(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    train, test = train_test_split_by_year(features)
    train.to_parquet(train_output_path, index=False)
    test.to_parquet(test_output_path, index=False)

    return train, test


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    train, test = build_feature_table(
        clean_data_path=project_root / "data" / "raw" / "ev_clean.parquet",
        output_path=project_root / "data" / "processed" / "features.parquet",
        train_output_path=project_root / "data" / "processed" / "features_train.parquet",
        test_output_path=project_root / "data" / "processed" / "features_test.parquet",
    )
    print(f"Train set: {train.shape}, years {train['year'].min()}-{train['year'].max()}")
    print(f"Test set: {test.shape}, years {test['year'].min()}-{test['year'].max()}")
