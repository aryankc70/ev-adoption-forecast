"""Shared feature preparation for gradient boosting model candidates."""

import pandas as pd

NUMERIC_FEATURES = [
    "year",
    "ev_sales_share_lag1",
    "ev_sales_share_yoy_change",
    "takeoff_year",
    "years_since_takeoff",
    "cumulative_years_tracked",
]
CATEGORICAL_FEATURES = ["region_country", "mode"]
TARGET_COL = "ev_sales_share"
WEIGHT_COL = "sample_weight"


def prepare_gbm_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Build aligned, one-hot encoded feature matrices for train and test sets.

    Returns (X_train, y_train, weights_train, X_test, y_test, test_meta), where
    test_meta preserves the original identifying columns (region_country, mode,
    year) alongside the test set, since those are lost once features are one-hot
    encoded but are needed later to join predictions back for evaluation.

    Rows with missing lag features (i.e. a country/mode's first tracked year,
    which has no prior-year value) are dropped -- a GBM cannot use NaN inputs
    the way the logistic baseline's pure time-based fit could.
    """
    train_clean = train_df.dropna(subset=NUMERIC_FEATURES + [TARGET_COL]).copy()
    test_clean = test_df.dropna(subset=NUMERIC_FEATURES + [TARGET_COL]).copy()

    test_meta = test_clean[["region_country", "mode", "year"]].reset_index(drop=True)

    combined = pd.concat(
        [train_clean.assign(_split="train"), test_clean.assign(_split="test")],
        ignore_index=True,
    )

    # One-hot encode categoricals across the combined data so train/test end up
    # with identical columns, even if a category only appears in one split.
    encoded = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)

    train_encoded = encoded[encoded["_split"] == "train"].drop(columns=["_split"])
    test_encoded = encoded[encoded["_split"] == "test"].drop(columns=["_split"])

    feature_cols = [
        c for c in encoded.columns if c not in {"_split", TARGET_COL, WEIGHT_COL, "ev_sales_volume"}
    ]

    train_meta = train_clean[
        [
            "region_country",
            "mode",
            "year",
            "ev_sales_share",
            "takeoff_year",
            "cumulative_years_tracked",
        ]
    ].reset_index(drop=True)

    x_train = train_encoded[feature_cols].reset_index(drop=True)
    y_train = train_encoded[TARGET_COL].reset_index(drop=True)
    weights_train = train_encoded[WEIGHT_COL].reset_index(drop=True)

    x_test = test_encoded[feature_cols].reset_index(drop=True)
    y_test = test_encoded[TARGET_COL].reset_index(drop=True)

    return x_train, y_train, weights_train, x_test, y_test, test_meta, train_meta
