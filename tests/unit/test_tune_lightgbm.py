"""Unit tests for time-series CV fold construction in LightGBM tuning."""

import pandas as pd

from ev_forecast.models.tune_lightgbm import _make_cv_fold


class TestMakeCvFold:
    def test_train_fold_excludes_years_after_cutoff(self) -> None:
        df = pd.DataFrame({"year": [2017, 2018, 2019, 2020, 2021]})
        cv_train, _ = _make_cv_fold(df, cutoff=2018, horizon=2)
        assert set(cv_train["year"]) == {2017, 2018}

    def test_val_fold_is_strictly_after_cutoff_within_horizon(self) -> None:
        df = pd.DataFrame({"year": [2017, 2018, 2019, 2020, 2021]})
        _, cv_val = _make_cv_fold(df, cutoff=2018, horizon=2)
        assert set(cv_val["year"]) == {2019, 2020}

    def test_train_and_val_folds_never_overlap(self) -> None:
        df = pd.DataFrame({"year": [2017, 2018, 2019, 2020, 2021]})
        cv_train, cv_val = _make_cv_fold(df, cutoff=2019, horizon=2)
        assert set(cv_train["year"]) & set(cv_val["year"]) == set()

    def test_val_fold_never_reaches_beyond_horizon(self) -> None:
        df = pd.DataFrame({"year": [2017, 2018, 2019, 2020, 2021, 2022]})
        _, cv_val = _make_cv_fold(df, cutoff=2019, horizon=1)
        assert 2021 not in set(cv_val["year"])
        assert set(cv_val["year"]) == {2020}
