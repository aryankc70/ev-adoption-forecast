"""Integration test: raw Excel-shaped data -> features -> baseline model, end to end.

Unlike the unit tests (which test each function in isolation with minimal synthetic
data), this test exercises the real module boundaries together -- loader output
feeding correctly into feature engineering, feature engineering output feeding
correctly into the model -- using a slightly larger synthetic dataset shaped like
the real IEA export. It does not touch the real 2.5MB Excel file (that stays a
manual, non-CI-friendly artifact), but it does touch every module's real, non-mocked
code path.
"""

from pathlib import Path

import pandas as pd
import pytest

from ev_forecast.data.loader import load_and_clean
from ev_forecast.features.build_features import build_feature_table
from ev_forecast.models.baseline import fit_all_curves, predict_with_curves
from ev_forecast.models.evaluation import evaluate_predictions


def _make_synthetic_iea_excel(path: Path) -> None:
    """Build a multi-year, multi-country synthetic Excel file shaped like the real IEA export."""
    years = list(range(2015, 2026))
    rows = []
    for country, base, growth in [("Testland", 0.5, 1.3), ("Otherland", 2.0, 1.15)]:
        share = base
        for year in years:
            rows.append(
                {
                    "region_country": country,
                    "category": "Historical",
                    "parameter": "EV sales share",
                    "mode": "Cars",
                    "powertrain": "EV",
                    "year": year,
                    "unit": "%",
                    "value": round(min(share, 95.0), 3),
                    "Aggregate group": None,
                }
            )
            rows.append(
                {
                    "region_country": country,
                    "category": "Historical",
                    "parameter": "EV sales",
                    "mode": "Cars",
                    "powertrain": "EV",
                    "year": year,
                    "unit": "Vehicles",
                    "value": round(share * 1000, 1),
                    "Aggregate group": None,
                }
            )
            share *= growth

    main_df = pd.DataFrame(rows)
    regions_df = pd.DataFrame(
        [
            {"region_country": "World", "Agg_group": "_World"},
            {"region_country": "Europe", "Agg_group": "Other_aggregate"},
        ]
    )

    with pd.ExcelWriter(path) as writer:
        main_df.to_excel(writer, sheet_name="GEVO_EV_2026", index=False)
        regions_df.to_excel(writer, sheet_name="Regions and countries", index=False)


@pytest.fixture
def synthetic_pipeline_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "excel": tmp_path / "synthetic.xlsx",
        "clean_parquet": tmp_path / "clean.parquet",
        "manifest": tmp_path / "manifest.json",
        "features_parquet": tmp_path / "features.parquet",
        "train_parquet": tmp_path / "features_train.parquet",
        "test_parquet": tmp_path / "features_test.parquet",
    }


class TestFullPipeline:
    def test_loader_to_features_to_model_end_to_end(
        self, synthetic_pipeline_paths: dict[str, Path]
    ) -> None:
        paths = synthetic_pipeline_paths
        _make_synthetic_iea_excel(paths["excel"])

        # Stage 1: load and clean
        load_report = load_and_clean(paths["excel"], paths["clean_parquet"], paths["manifest"])
        assert load_report.rows_after_target_mode_filter > 0
        assert paths["clean_parquet"].exists()

        # Stage 2: build features (also performs the train/test split)
        train, test = build_feature_table(
            paths["clean_parquet"],
            paths["features_parquet"],
            paths["train_parquet"],
            paths["test_parquet"],
        )
        assert len(train) > 0
        assert len(test) > 0
        assert train["year"].max() < test["year"].min()  # no chronological overlap

        # Stage 3: fit the baseline model on the resulting train set
        fits = fit_all_curves(train)
        assert len(fits) > 0
        assert any(f.converged for f in fits.values())

        # Stage 4: predict on the resulting test set and evaluate
        predictions = predict_with_curves(test, fits)
        predictions_clean = predictions.dropna(subset=["predicted_share"])
        assert len(predictions_clean) > 0

        result = evaluate_predictions(predictions_clean)
        assert result.n_observations > 0
        assert result.mape >= 0  # sanity: a real, non-negative metric was computed

    def test_pipeline_excludes_aggregate_regions_end_to_end(
        self, synthetic_pipeline_paths: dict[str, Path]
    ) -> None:
        """The World/Europe aggregate rows in the synthetic file must never reach the model."""
        paths = synthetic_pipeline_paths
        _make_synthetic_iea_excel(paths["excel"])

        load_and_clean(paths["excel"], paths["clean_parquet"], paths["manifest"])
        train, test = build_feature_table(
            paths["clean_parquet"],
            paths["features_parquet"],
            paths["train_parquet"],
            paths["test_parquet"],
        )

        all_countries = set(train["region_country"]) | set(test["region_country"])
        assert "World" not in all_countries
        assert "Europe" not in all_countries
