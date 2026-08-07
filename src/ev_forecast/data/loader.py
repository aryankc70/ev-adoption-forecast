"""Load, validate, and clean raw IEA Global EV Outlook data."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from ev_forecast.data.schemas import (
    TARGET_MODES,
    DataCategory,
    RawEVRecord,
    RegionClassification,
)

RAW_DATA_SHEET = "GEVO_EV_2026"
REGIONS_SHEET = "Regions and countries"


@dataclass
class LoadReport:
    """Summary of a data load run, used to build the manifest and catch silent data loss."""

    source_file: str
    source_sha256: str
    loaded_at: str
    total_rows_in_source: int
    rows_failed_validation: int
    rows_after_country_filter: int
    rows_after_target_mode_filter: int
    countries_included: list[str] = field(default_factory=list)
    validation_errors_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def _file_sha256(path: Path) -> str:
    """Hash the source file so the manifest proves exactly which snapshot was used."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_region_classifications(excel_path: Path) -> dict[str, RegionClassification]:
    """Read the 'Regions and countries' sheet and return a lookup by region_country name."""
    df = pd.read_excel(excel_path, sheet_name=REGIONS_SHEET)
    lookup: dict[str, RegionClassification] = {}
    for row in df.to_dict(orient="records"):
        rc = RegionClassification.model_validate(row)
        lookup[rc.region_country] = rc
    return lookup


def load_and_clean(
    excel_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> LoadReport:
    """
    Load the raw IEA Excel export, validate every row, filter to country-level
    historical records for our target vehicle modes (Cars, 2/3-wheelers), and
    write the cleaned result to Parquet plus a JSON manifest documenting the run.
    """
    raw_df = pd.read_excel(excel_path, sheet_name=RAW_DATA_SHEET)
    region_lookup = _load_region_classifications(excel_path)

    validated_records: list[RawEVRecord] = []
    validation_errors: list[str] = []
    failed_rows_in_target_modes = 0

    for row in raw_df.to_dict(orient="records"):
        try:
            record = RawEVRecord.model_validate(row)
            validated_records.append(record)
        except ValidationError as e:
            validation_errors.append(str(e))
            if row.get("mode") in {m.value for m in TARGET_MODES}:
                failed_rows_in_target_modes += 1

    if failed_rows_in_target_modes > 0:
        raise ValueError(
            f"{failed_rows_in_target_modes} rows in target modes (Cars / "
            "2 and 3 wheelers) failed validation. This is a real data quality "
            "issue, not noise -- investigate before proceeding."
        )

    # The lookup sheet only lists SPECIAL cases: aggregate/rollup regions,
    # plus a handful of countries IEA also tags for projection modeling.
    # Any region_country NOT in this lookup is a normal standalone country
    # and should be included by default.
    country_records = [
        r
        for r in validated_records
        if region_lookup.get(r.region_country) is None or region_lookup[r.region_country].is_country
    ]

    target_records = [
        r
        for r in country_records
        if r.category == DataCategory.HISTORICAL and r.mode in {m.value for m in TARGET_MODES}
    ]

    clean_df = pd.DataFrame([r.model_dump() for r in target_records])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path, index=False)

    report = LoadReport(
        source_file=str(excel_path),
        source_sha256=_file_sha256(excel_path),
        loaded_at=datetime.now(UTC).isoformat(),
        total_rows_in_source=len(raw_df),
        rows_failed_validation=len(validation_errors),
        rows_after_country_filter=len(country_records),
        rows_after_target_mode_filter=len(target_records),
        countries_included=sorted({r.region_country for r in target_records}),
        validation_errors_sample=validation_errors[:5],
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report.to_dict(), indent=2))

    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    report = load_and_clean(
        excel_path=project_root / "data" / "external" / "ev_data_by_country_2026.xlsx",
        output_path=project_root / "data" / "raw" / "ev_clean.parquet",
        manifest_path=project_root / "data" / "raw" / "load_manifest.json",
    )
    print(f"Loaded {report.rows_after_target_mode_filter} clean rows")
    print(f"Countries: {len(report.countries_included)}")
    print(f"Validation failures: {report.rows_failed_validation}")
