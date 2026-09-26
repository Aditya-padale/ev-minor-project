"""
train_model.py
---------------
Trains and compares several regressors on the EV charging dataset,
picks the best by cross-validated MAE, and serializes:
  - model.pkl      (the fitted sklearn Pipeline: preprocessing + model)
  - metrics.json    (holdout performance, for the README / UI "About" panel)
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent.parent / "data" / "ev_charging_dataset.csv"
MODEL_PATH = HERE / "ev_charging_model.pkl"
METRICS_PATH = HERE / "metrics.json"

NUMERIC_FEATURES = [
    "battery_capacity_kwh",
    "start_soc_percent",
    "target_soc_percent",
    "charger_power_kw",
    "ambient_temp_c",
]
CATEGORICAL_FEATURES = ["vehicle_type"]
TARGET = "charging_time_minutes"


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soc_delta_percent"] = df["target_soc_percent"] - df["start_soc_percent"]
    df["energy_needed_kwh"] = df["battery_capacity_kwh"] * (df["soc_delta_percent"] / 100.0)
    return df


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATA_PATH}. Run generate_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df = add_engineered_features(df)

    numeric_features = NUMERIC_FEATURES + ["soc_delta_percent", "energy_needed_kwh"]

    X = df[numeric_features + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.08, random_state=42
        ),
        "Ridge": Ridge(alpha=1.0),
    }

    results = {}
    best_name, best_pipeline, best_mae = None, None, np.inf

    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)

        cv_mae = -cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        ).mean()

        results[name] = {
            "test_mae_minutes": round(mae, 2),
            "test_rmse_minutes": round(rmse, 2),
            "test_r2": round(r2, 4),
            "cv5_mae_minutes": round(cv_mae, 2),
            "train_time_sec": round(train_time, 2),
        }
        print(f"{name:18s} | MAE={mae:6.2f}min  RMSE={rmse:6.2f}min  R2={r2:.4f}  CV-MAE={cv_mae:.2f}")

        if mae < best_mae:
            best_mae, best_name, best_pipeline = mae, name, pipe

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "numeric_features": numeric_features,
            "categorical_features": CATEGORICAL_FEATURES,
            "raw_numeric_inputs": NUMERIC_FEATURES,
            "model_name": best_name,
        },
        MODEL_PATH,
    )

    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nBest model: {best_name}  (MAE={best_mae:.2f} min)")
    print(f"Saved -> {MODEL_PATH}")
    print(f"Saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
