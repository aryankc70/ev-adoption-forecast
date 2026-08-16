"""FastAPI serving app for EV adoption share forecasts."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from ev_forecast.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "production"

_model: Any = None
_feature_columns: list[str] = []
_group_states: dict[tuple[str, str], dict] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the trained model and its supporting artifacts once, at process startup."""
    global _model, _feature_columns, _group_states

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found at {MODEL_DIR}. Run "
            "`python -m ev_forecast.pipelines.train_final_model` first."
        )

    _model = joblib.load(MODEL_DIR / "model.joblib")
    _feature_columns = json.loads((MODEL_DIR / "feature_columns.json").read_text())

    states_raw = json.loads((MODEL_DIR / "group_states.json").read_text())
    _group_states = {(s["region_country"], s["mode"]): s for s in states_raw}

    yield


app = FastAPI(
    title="EV Adoption Forecast API",
    description="Forecasts EV sales share by country and vehicle mode using a tuned LightGBM model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Forecast EV sales share for a given country/mode/year.

    For year 2022 (the first forecast year), this uses the country's real last
    known training-period state directly. For later years, a true API would need
    to recursively step through every intermediate year first (exactly like
    recursive_forecast.py), since each year's lag features depend on the
    previous year's prediction. This endpoint performs that same recursive
    walk internally, so the caller can request any single target year directly.
    """
    key = (request.region_country, request.mode)
    state = _group_states.get(key)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model state for region_country='{request.region_country}', "
            f"mode='{request.mode}'. Check spelling and known values.",
        )

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    last_share = state["last_share"]
    prior_share = state["prior_share"]
    years_tracked = state["cumulative_years_tracked"]
    takeoff_year = state["takeoff_year"]
    template_row = state["template_row"]

    # Recursively walk forward from the training boundary up to the requested
    # year, since each step's lag features depend on the previous step's
    # prediction -- identical logic to recursive_forecast.py, applied live here.
    training_last_year = int(state["last_known_year"])
    target_years = range(training_last_year + 1, request.year + 1)

    predicted_share = last_share
    for year in target_years:
        years_tracked += 1
        yoy_change = (last_share - prior_share) if prior_share is not None else np.nan
        years_since_takeoff = (year - takeoff_year) if takeoff_year is not None else np.nan

        feature_row = dict(template_row)
        feature_row["year"] = float(year)
        feature_row["ev_sales_share_lag1"] = last_share
        feature_row["ev_sales_share_yoy_change"] = yoy_change
        feature_row["takeoff_year"] = takeoff_year if takeoff_year is not None else np.nan
        feature_row["years_since_takeoff"] = years_since_takeoff
        feature_row["cumulative_years_tracked"] = years_tracked

        x_row = pd.DataFrame([feature_row])[_feature_columns].fillna(0.0)
        predicted_share = float(_model.predict(x_row)[0])

        prior_share = last_share
        last_share = predicted_share

    return PredictionResponse(
        region_country=request.region_country,
        mode=request.mode,
        year=request.year,
        predicted_share=predicted_share,
    )
