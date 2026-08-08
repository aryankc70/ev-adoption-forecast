"""Build model-ready features from cleaned EV adoption data."""

from pathlib import Path

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


def build_feature_table(clean_data_path: Path, output_path: Path) -> pd.DataFrame:
    """Full feature engineering pipeline: clean parquet -> model-ready feature table."""
    df = pd.read_parquet(clean_data_path)

    features = _pivot_target(df)
    features = _add_lag_features(features)
    features = _add_takeoff_features(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    return features


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    result = build_feature_table(
        clean_data_path=project_root / "data" / "raw" / "ev_clean.parquet",
        output_path=project_root / "data" / "processed" / "features.parquet",
    )
    print(f"Built feature table: {result.shape}")
    print(result.head(10).to_string())
