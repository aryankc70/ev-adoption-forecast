"""Unit tests for the data loading and validation pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from ev_forecast.data.loader import _load_region_classifications, load_and_clean
from ev_forecast.data.schemas import RawEVRecord, RegionClassification


class TestRawEVRecord:
    def test_valid_row_parses(self) -> None:
        record = RawEVRecord.model_validate(
            {
                "region_country": "Norway",
                "category": "Historical",
                "parameter": "EV sales share",
                "mode": "Cars",
                "powertrain": "EV",
                "year": 2024,
                "unit": "%",
                "value": 92.0,
                "Aggregate group": None,
            }
        )
        assert record.region_country == "Norway"
        assert record.value == 92.0

    def test_nan_unit_becomes_none(self) -> None:
        record = RawEVRecord.model_validate(
            {
                "region_country": "Norway",
                "category": "Historical",
                "parameter": "EV sales",
                "mode": "Cars",
                "powertrain": "EV",
                "year": 2024,
                "unit": float("nan"),
                "value": 100.0,
            }
        )
        assert record.unit is None

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            RawEVRecord.model_validate(
                {
                    "region_country": "Norway",
                    "category": "Historical",
                    "parameter": "EV sales share",
                    "mode": "Cars",
                    "powertrain": "EV",
                    "year": 2024,
                    "value": -5.0,
                }
            )

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValueError):
            RawEVRecord.model_validate(
                {
                    "region_country": "Norway",
                    "category": "NotARealCategory",
                    "parameter": "EV sales share",
                    "mode": "Cars",
                    "powertrain": "EV",
                    "year": 2024,
                    "value": 5.0,
                }
            )


class TestRegionClassification:
    def test_aggregate_region_is_not_country(self) -> None:
        rc = RegionClassification.model_validate(
            {"region_country": "Europe", "Agg_group": "Other_aggregate"}
        )
        assert rc.is_country is False

    def test_projection_country_is_still_a_country(self) -> None:
        rc = RegionClassification.model_validate(
            {"region_country": "China", "Agg_group": "Projection_country"}
        )
        assert rc.is_country is True

    def test_world_is_not_country(self) -> None:
        rc = RegionClassification.model_validate({"region_country": "World", "Agg_group": "_World"})
        assert rc.is_country is False


@pytest.fixture
def synthetic_excel(tmp_path: Path) -> Path:
    """Build a small in-memory Excel file mimicking the real IEA structure."""
    excel_path = tmp_path / "synthetic_ev_data.xlsx"

    main_df = pd.DataFrame(
        [
            # A normal country, target mode, historical -> should survive
            {
                "region_country": "Norway",
                "category": "Historical",
                "parameter": "EV sales share",
                "mode": "Cars",
                "powertrain": "EV",
                "year": 2024,
                "unit": "%",
                "value": 92.0,
                "Aggregate group": None,
            },
            # An aggregate region -> should be excluded
            {
                "region_country": "Europe",
                "category": "Historical",
                "parameter": "EV sales share",
                "mode": "Cars",
                "powertrain": "EV",
                "year": 2024,
                "unit": "%",
                "value": 40.0,
                "Aggregate group": "Other",
            },
            # A projection row -> should be excluded (not Historical)
            {
                "region_country": "Norway",
                "category": "Projection-STEPS",
                "parameter": "EV sales share",
                "mode": "Cars",
                "powertrain": "EV",
                "year": 2030,
                "unit": "%",
                "value": 99.0,
                "Aggregate group": None,
            },
            # A non-target mode -> should be excluded
            {
                "region_country": "Norway",
                "category": "Historical",
                "parameter": "EV sales share",
                "mode": "Buses",
                "powertrain": "EV",
                "year": 2024,
                "unit": "%",
                "value": 10.0,
                "Aggregate group": None,
            },
        ]
    )

    regions_df = pd.DataFrame(
        [
            {"region_country": "World", "Agg_group": "_World"},
            {"region_country": "Europe", "Agg_group": "Other_aggregate"},
            {"region_country": "China", "Agg_group": "Projection_country"},
        ]
    )

    with pd.ExcelWriter(excel_path) as writer:
        main_df.to_excel(writer, sheet_name="GEVO_EV_2026", index=False)
        regions_df.to_excel(writer, sheet_name="Regions and countries", index=False)

    return excel_path


class TestLoadAndClean:
    def test_filters_to_historical_target_country_rows_only(
        self, synthetic_excel: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "clean.parquet"
        manifest_path = tmp_path / "manifest.json"

        report = load_and_clean(synthetic_excel, output_path, manifest_path)

        # Only the first row (Norway, Cars, Historical, EV) should survive
        assert report.rows_after_target_mode_filter == 1
        assert report.countries_included == ["Norway"]

        clean_df = pd.read_parquet(output_path)
        assert len(clean_df) == 1
        assert clean_df.iloc[0]["region_country"] == "Norway"
        assert clean_df.iloc[0]["value"] == 92.0

    def test_manifest_file_is_written(self, synthetic_excel: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "clean.parquet"
        manifest_path = tmp_path / "manifest.json"

        load_and_clean(synthetic_excel, output_path, manifest_path)

        assert manifest_path.exists()

    def test_region_classifications_load_correctly(self, synthetic_excel: Path) -> None:
        lookup = _load_region_classifications(synthetic_excel)
        assert lookup["World"].is_country is False
        assert lookup["China"].is_country is True
