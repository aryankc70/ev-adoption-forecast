"""Pydantic schemas for validating raw IEA Global EV Outlook data."""

from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class AggregateGroup(StrEnum):
    """Classification of a region_country value, per IEA's own lookup sheet."""

    WORLD = "_World"
    OTHER_AGGREGATE = "Other_aggregate"
    PROJECTION_REGION = "Projection_region"
    PROJECTION_COUNTRY = "Projection_country"
    EUROPEAN_UNION = "European Union"


# Values that represent rollups/aggregates rather than standalone countries.
# Rows with these classifications must be excluded from country-level modeling
# to avoid double-counting (e.g. "Europe" already sums its member countries).
NON_COUNTRY_GROUPS = {
    AggregateGroup.WORLD,
    AggregateGroup.OTHER_AGGREGATE,
    AggregateGroup.PROJECTION_REGION,
    AggregateGroup.EUROPEAN_UNION,
}


class DataCategory(StrEnum):
    HISTORICAL = "Historical"
    PROJECTION_CPS = "Projection-CPS"
    PROJECTION_STEPS = "Projection-STEPS"


class VehicleMode(StrEnum):
    CARS = "Cars"
    TWO_THREE_WHEELERS = "2 and 3 wheelers"
    BUSES = "Buses"
    TRUCKS = "Trucks"
    VANS = "Vans"
    EVSE = "EVSE"
    EV = "EV"
    CITY_CAR = "City car"
    LARGE_CAR = "Large car"
    MEDIUM_CAR = "Medium car"
    SMALL_SUV_PICKUP = "Small SUV/Pick-up"
    LARGE_SUV_PICKUP = "Large SUV/Pick-up"
    ALL_CARS = "All cars"


# The two vehicle modes this project targets.
TARGET_MODES = {VehicleMode.CARS, VehicleMode.TWO_THREE_WHEELERS}


class RawEVRecord(BaseModel):
    """A single validated row from the IEA GEVO_EV_2026 sheet."""

    region_country: str = Field(min_length=1)
    category: DataCategory
    parameter: str
    mode: str
    powertrain: str
    year: int = Field(ge=2000, le=2050)
    unit: str | None = None
    value: float
    aggregate_group: str | None = Field(default=None, alias="Aggregate group")

    @field_validator("unit", "aggregate_group", mode="before")
    @classmethod
    def nan_to_none(cls, v: object) -> object:
        if isinstance(v, float) and pd.isna(v):
            return None
        return v

    @field_validator("value")
    @classmethod
    def value_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"value must be non-negative, got {v}")
        return v

    model_config = {"populate_by_name": True}


class RegionClassification(BaseModel):
    """A single row from the 'Regions and countries' lookup sheet."""

    region_country: str
    agg_group: AggregateGroup = Field(alias="Agg_group")

    model_config = {"populate_by_name": True}

    @property
    def is_country(self) -> bool:
        """True if this represents an actual standalone country, not a rollup region."""
        return self.agg_group not in NON_COUNTRY_GROUPS
