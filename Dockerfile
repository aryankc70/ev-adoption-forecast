# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# LightGBM's compiled core requires libgomp (GNU OpenMP runtime) for
# multi-threaded computation. python:3.12-slim deliberately omits it to
# stay minimal, so it must be installed explicitly -- a well-known,
# LightGBM-specific requirement on slim/minimal base images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry itself (separate layer, changes rarely -> cached across builds)
RUN pip install "poetry==2.4.1"

# Copy only dependency manifests first, so Docker's layer cache is reused
# whenever source code changes but dependencies don't -- avoids reinstalling
# every package on every single code edit during development.
COPY pyproject.toml poetry.lock ./

# --only main: skip dev-only tooling (pytest, black, mypy, pre-commit) --
# none of that belongs in a production serving image.
RUN poetry install --only main --no-root

# Now copy the actual application code and pre-trained model artifacts.
# The model is baked into the image at build time (not trained at runtime,
# not volume-mounted) -- a deliberate simplicity/reproducibility tradeoff for
# this project's scale. A real production system with frequently-updated
# models would more likely load from a model registry or mounted volume instead.
COPY src ./src
COPY models/production ./models/production

# Install the project's own package (editable install skipped above via --no-root,
# now needed so `ev_forecast` is importable)
RUN poetry install --only main

EXPOSE 8000

CMD ["uvicorn", "ev_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
