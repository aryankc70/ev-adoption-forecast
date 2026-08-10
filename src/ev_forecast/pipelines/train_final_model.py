"""Train and persist the final production model (tuned LightGBM) plus its supporting artifacts."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from ev_forecast.models.gbm_features import prepare_gbm_features
from ev_forecast.models.recursive_forecast import _build_group_states

TUNED_LIGHTGBM_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 7,
    "num_leaves": 15,
}
RANDOM_SEED = 42


def train_and_save_final_model(train_path: Path, test_path: Path, output_dir: Path) -> None:
    """
    Train the selected production model on the full training set and persist
    everything the serving API needs: the fitted model, the exact feature column
    order (one-hot encoding must match at inference time), and each country/mode's
    last known state (needed to build the lag/trend features for a genuinely new
    forecast, the same way recursive_forecast.py does).
    """
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    x_train, y_train, w_train, _, _, _, train_meta = prepare_gbm_features(train, test)

    model = lgb.LGBMRegressor(
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_estimators=int(TUNED_LIGHTGBM_PARAMS["n_estimators"]),
        learning_rate=TUNED_LIGHTGBM_PARAMS["learning_rate"],
        max_depth=int(TUNED_LIGHTGBM_PARAMS["max_depth"]),
        num_leaves=int(TUNED_LIGHTGBM_PARAMS["num_leaves"]),
    )
    model.fit(x_train, y_train, sample_weight=w_train)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")

    feature_columns = list(x_train.columns)
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))

    last_years = train_meta.groupby(["region_country", "mode"])["year"].max()

    states = _build_group_states(x_train, train_meta)
    states_serializable = [
        {
            "region_country": s.region_country,
            "mode": s.mode,
            "last_known_year": int(last_years[(s.region_country, s.mode)]),
            "last_share": s.last_share,
            "prior_share": s.prior_share,
            "takeoff_year": s.takeoff_year,
            "cumulative_years_tracked": s.cumulative_years_tracked,
            "template_row": s.template_row.to_dict(),
        }
        for s in states
    ]
    (output_dir / "group_states.json").write_text(json.dumps(states_serializable, indent=2))

    print(
        f"Saved model, {len(feature_columns)} feature columns, and {len(states)} group states to {output_dir}"
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    train_and_save_final_model(
        train_path=project_root / "data" / "processed" / "features_train.parquet",
        test_path=project_root / "data" / "processed" / "features_test.parquet",
        output_dir=project_root / "models" / "production",
    )
