"""Request/response schemas for the EV adoption forecast API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """A request to forecast EV sales share for one country/mode/year."""

    region_country: str = Field(description="Country name, e.g. 'Germany'")
    mode: str = Field(description="'Cars' or '2 and 3 wheelers'")
    year: int = Field(ge=2022, le=2035, description="Forecast target year")


class PredictionResponse(BaseModel):
    """A single forecast result."""

    region_country: str
    mode: str
    year: int
    predicted_share: float = Field(description="Predicted EV sales share, percent")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
