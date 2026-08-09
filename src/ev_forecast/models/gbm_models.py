"""Gradient boosting model candidates: LightGBM, XGBoost, and sklearn GBM."""

from dataclasses import dataclass
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_SEED = 42


class FittedRegressor(Protocol):
    """Structural type for any of the three fitted model objects we use."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ModelCandidate:
    """A named, fitted model ready for prediction."""

    name: str
    model: FittedRegressor


def train_lightgbm(
    x_train: pd.DataFrame, y_train: pd.Series, weights_train: pd.Series
) -> ModelCandidate:
    model = lgb.LGBMRegressor(
        random_state=RANDOM_SEED,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=weights_train)
    return ModelCandidate(name="LightGBM", model=model)


def train_xgboost(
    x_train: pd.DataFrame, y_train: pd.Series, weights_train: pd.Series
) -> ModelCandidate:
    model = xgb.XGBRegressor(
        random_state=RANDOM_SEED,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
    )
    model.fit(x_train, y_train, sample_weight=weights_train)
    return ModelCandidate(name="XGBoost", model=model)


def train_sklearn_gbm(
    x_train: pd.DataFrame, y_train: pd.Series, weights_train: pd.Series
) -> ModelCandidate:
    model = GradientBoostingRegressor(
        random_state=RANDOM_SEED,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
    )
    model.fit(x_train, y_train, sample_weight=weights_train)
    return ModelCandidate(name="sklearn GBM", model=model)


def train_all_candidates(
    x_train: pd.DataFrame, y_train: pd.Series, weights_train: pd.Series
) -> list[ModelCandidate]:
    """Train all three GBM candidates with identical hyperparameters, for a fair comparison."""
    return [
        train_lightgbm(x_train, y_train, weights_train),
        train_xgboost(x_train, y_train, weights_train),
        train_sklearn_gbm(x_train, y_train, weights_train),
    ]
