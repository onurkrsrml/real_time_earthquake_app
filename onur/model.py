"""Model training and inference utilities for the earthquake prediction app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


DATA_PATH = Path("data/earthquakes_featured.csv")
ARTIFACT_DIR = Path("onur/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_MAG_PATH = ARTIFACT_DIR / "model_mag.joblib"
MODEL_DAYS_PATH = ARTIFACT_DIR / "model_days.joblib"
FEATURES_PATH = ARTIFACT_DIR / "features.json"

MAJOR_MAG = 4.0


@dataclass
class TrainingResult:
    model_mag: RandomForestRegressor
    model_days: RandomForestRegressor
    features: list[str]


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def _safe_rolling(group: pd.DataFrame, col: str, window: int, func: str = "mean") -> pd.Series:
    shifted = group[col].shift(1)
    if func == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    if func == "max":
        return shifted.rolling(window, min_periods=1).max()
    if func == "std":
        return shifted.rolling(window, min_periods=1).std()
    if func == "sum":
        return shifted.rolling(window, min_periods=1).sum()
    return shifted.rolling(window, min_periods=1).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "id" in df.columns:
        df = df.drop(["id"], axis=1)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)

    column_aliases = {
        "time": "date",
        "depth_km": "depth",
        "solar_flux_f107": "solar_flux",
        "b_value_trend": "b_value",
    }
    df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns}, inplace=True)

    if "region" not in df.columns:
        df["region"] = "unknown"

    if "grid_id" not in df.columns:
        df["grid_id"] = df["region"].astype(str)

    df = df.sort_values(by=["date"]).reset_index(drop=True)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [col for col in num_cols if col not in {"magnitude", "depth"}]

    df.set_index("date", inplace=True)
    if num_cols:
        df[num_cols] = df[num_cols].interpolate(method="time").bfill()
    df.reset_index(inplace=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    df["season"] = df["month"] % 12 // 3 + 1
    df["days_since_last_event"] = df.groupby("grid_id")["date"].diff().dt.days

    df["days_since_last_major_event"] = df.groupby("grid_id").apply(
        lambda g: g["date"].where(g["magnitude"] >= MAJOR_MAG).ffill()
    ).reset_index(level=0, drop=True)
    df["days_since_last_major_event"] = (df["date"] - df["days_since_last_major_event"]).dt.days

    if "event_count" not in df.columns:
        df["event_count"] = 1

    for w in [7, 14, 30]:
        df[f"event_frequency_last_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
            lambda g: _safe_rolling(g, "event_count", w, func="sum")
        )

    for w in [7, 14, 30]:
        df[f"rolling_mean_magnitude_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
            lambda g: _safe_rolling(g, "magnitude", w, func="mean")
        )
        df[f"rolling_max_magnitude_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
            lambda g: _safe_rolling(g, "magnitude", w, func="max")
        )

    df["magnitude_std_7d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g, "magnitude", 7, func="std")
    )

    df["magnitude_acceleration"] = df["rolling_mean_magnitude_7d"] - df["rolling_mean_magnitude_14d"]
    df["magnitude_trend"] = df["rolling_mean_magnitude_7d"] - df["rolling_mean_magnitude_30d"]

    df["energy"] = 10 ** (1.5 * df["magnitude"] + 4.8)
    for w in [7, 14, 30]:
        df[f"rolling_energy_sum_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
            lambda g: _safe_rolling(g, "energy", w, func="sum")
        )

    df["energy_acceleration"] = df["rolling_energy_sum_7d"] - df["rolling_energy_sum_14d"]
    df["energy_release_rate"] = df["rolling_energy_sum_7d"] / 7

    energy_mean_30 = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g, "energy", 30, func="mean")
    )
    energy_std_30 = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g, "energy", 30, func="std")
    )
    df["energy_anomaly_score"] = (df["energy"] - energy_mean_30) / (energy_std_30 + 1e-6)

    for w in [7, 14, 30]:
        df[f"event_count_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
            lambda g: _safe_rolling(g, "event_count", w, func="sum")
        )

    df["microquake_density"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g.assign(micro=(g["magnitude"] < 2.5).astype(int)), "micro", 30, func="sum")
    )

    df["foreshock_density"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(
            g.assign(foreshock=((g["magnitude"] >= 3.0) & (g["magnitude"] < 4.0)).astype(int)),
            "foreshock",
            30,
            func="sum",
        )
    )

    df["cluster_intensity"] = df["event_count_7d"] / (df["event_count_30d"] + 1)
    df["seismic_burst_score"] = df["event_count_7d"] / (df["event_count_14d"] + 1)

    df["rolling_mean_depth_7d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g, "depth", 7, func="mean")
    )
    df["rolling_mean_depth_30d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g, "depth", 30, func="mean")
    )

    df["depth_std_30d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g, "depth", 30, func="std")
    )

    df["shallow_event_ratio"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g.assign(shallow=(g["depth"] <= 70).astype(int)), "shallow", 30, func="mean")
    )
    df["deep_event_ratio"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: _safe_rolling(g.assign(deep=(g["depth"] >= 300).astype(int)), "deep", 30, func="mean")
    )

    if "country" in df.columns:
        df["regional_event_density"] = df.groupby("country", group_keys=False).apply(
            lambda g: _safe_rolling(g, "event_count", 30, func="sum")
        )
    else:
        df["regional_event_density"] = df["event_count_30d"]

    df["neighbor_grid_activity"] = df["regional_event_density"]

    df["magnitude_depth_ratio"] = df["magnitude"] / (df["depth"] + 1)
    df["rolling_mag_mean"] = df.groupby("region")["magnitude"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def compute_future_max_mag_7d(group: pd.DataFrame) -> pd.Series:
        times = group["date"].values
        mags = group["magnitude"].values
        result = np.zeros(len(group))
        for i in range(len(group)):
            future_mask = (times > times[i]) & (times <= times[i] + np.timedelta64(7, "D"))
            if future_mask.any():
                result[i] = mags[future_mask].max()
            else:
                result[i] = 0.0
        return pd.Series(result, index=group.index)

    def compute_days_to_next_major(group: pd.DataFrame) -> pd.Series:
        times = group["date"].values
        mags = group["magnitude"].values
        result = np.full(len(group), np.nan)
        for i in range(len(group)):
            future_mask = mags[i + 1 :] >= MAJOR_MAG
            if future_mask.any():
                next_idx = i + 1 + np.argmax(future_mask)
                delta_days = (times[next_idx] - times[i]) / np.timedelta64(1, "D")
                result[i] = delta_days
            else:
                result[i] = 60
        return pd.Series(result, index=group.index)

    df["future_max_magnitude_7d"] = df.groupby("grid_id", group_keys=False).apply(
        compute_future_max_mag_7d
    ).reset_index(level=0, drop=True)
    df["days_to_next_major_event"] = df.groupby("grid_id", group_keys=False).apply(
        compute_days_to_next_major
    ).reset_index(level=0, drop=True)
    df.dropna(subset=["future_max_magnitude_7d", "days_to_next_major_event"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = add_targets(df)
    drop_cols = ["date", "days_to_next_major_event", "future_max_magnitude_7d", "region", "grid_id"]
    features = [col for col in df.columns if col not in drop_cols]
    X = df[features]
    y_mag = df["future_max_magnitude_7d"]
    y_days = df["days_to_next_major_event"]
    return X, y_mag, y_days, features


def train_models(df: pd.DataFrame) -> TrainingResult:
    X, y_mag, y_days, features = prepare_training_data(df)

    tscv = TimeSeriesSplit(n_splits=5)
    model_mag = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model_days = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)

    for train_index, test_index in tscv.split(X):
        X_train, y_train = X.iloc[train_index], y_mag.iloc[train_index]
        model_mag.fit(X_train, y_train)

    for train_index, test_index in tscv.split(X):
        X_train, y_train = X.iloc[train_index], y_days.iloc[train_index]
        model_days.fit(X_train, y_train)

    model_mag.fit(X, y_mag)
    model_days.fit(X, y_days)

    return TrainingResult(model_mag=model_mag, model_days=model_days, features=features)


def save_artifacts(result: TrainingResult) -> None:
    joblib.dump(result.model_mag, MODEL_MAG_PATH)
    joblib.dump(result.model_days, MODEL_DAYS_PATH)
    with FEATURES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result.features, handle, indent=2)


def load_artifacts() -> tuple[RandomForestRegressor, RandomForestRegressor, list[str]]:
    model_mag = joblib.load(MODEL_MAG_PATH)
    model_days = joblib.load(MODEL_DAYS_PATH)
    with FEATURES_PATH.open("r", encoding="utf-8") as handle:
        features = json.load(handle)
    return model_mag, model_days, features


def ensure_models() -> tuple[RandomForestRegressor, RandomForestRegressor, list[str]]:
    if MODEL_MAG_PATH.exists() and MODEL_DAYS_PATH.exists() and FEATURES_PATH.exists():
        return load_artifacts()

    raw_df = load_raw_data()
    feature_df = build_features(raw_df)
    result = train_models(feature_df)
    save_artifacts(result)
    return result.model_mag, result.model_days, result.features


def predict_latest(df: pd.DataFrame) -> dict[str, float | str]:
    model_mag, model_days, features = ensure_models()
    X = df[features]
    latest_record = X.iloc[[-1]]

    mag_pred = float(model_mag.predict(latest_record)[0])
    days_pred = float(model_days.predict(latest_record)[0])

    output = {
        "prediction_date": date.today().strftime("%Y-%m-%d"),
        "days_to_event": round(days_pred, 1),
        "predicted_max_magnitude": round(mag_pred, 1),
    }
    return output


def predict_for_region(df: pd.DataFrame, region: str) -> dict[str, float | str]:
    model_mag, model_days, features = ensure_models()
    region_df = df[df["region"] == region]
    if region_df.empty:
        raise ValueError(f"Region not found: {region}")
    X = region_df[features]
    latest_record = X.iloc[[-1]]

    mag_pred = float(model_mag.predict(latest_record)[0])
    days_pred = float(model_days.predict(latest_record)[0])

    output = {
        "region": region,
        "prediction_date": date.today().strftime("%Y-%m-%d"),
        "days_to_event": round(days_pred, 1),
        "predicted_max_magnitude": round(mag_pred, 1),
    }
    return output


def load_featured_data() -> pd.DataFrame:
    raw_df = load_raw_data()
    feature_df = build_features(raw_df)
    return feature_df
