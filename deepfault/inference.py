"""Gerçek model yükleme ve canlı çıkarım."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from deepfault.analytics import aggregate_predictions_near, filter_events_by_date
from deepfault.features_rabia import build_feature_row_for_location
from deepfault.geo import filter_near
from deepfault.paths import (
    DATA_RAW,
    DEFAULT_RADIUS_KM,
    ONUR_METADATA,
    ONUR_MODEL_DAYS,
    ONUR_MODEL_MAG,
    PREDICTIONS_CSV,
    RABIA_MODEL,
)


@st.cache_data(show_spinner=False)
def load_raw_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["latitude", "longitude", "magnitude"])


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_rabia_bundle() -> dict:
    return joblib.load(RABIA_MODEL)


@st.cache_resource(show_spinner=False)
def load_onur_models() -> tuple[Any, Any]:
    return joblib.load(ONUR_MODEL_MAG), joblib.load(ONUR_MODEL_DAYS)


@st.cache_data(show_spinner=False)
def load_onur_metadata() -> dict:
    with open(ONUR_METADATA, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    from deepfault.paths import MODEL_METRICS_JSON

    with open(MODEL_METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _risk_level_from_prob(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "Düşük", "🟢"
    if prob < 0.60:
        return "Orta", "🟠"
    return "Yüksek", "🔴"


def run_rabia_inference(
    bundle: dict,
    events: pd.DataFrame,
    lat: float,
    lon: float,
    end_date: date,
) -> dict[str, Any]:
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    feature_row = build_feature_row_for_location(events, lat, lon, end_date=end_ts)
    if feature_row is None or feature_row.empty:
        return {"ok": False, "error": "Seçilen tarih aralığında bu konum için özellik üretilemedi."}

    features = bundle["features"]
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.5))

    X = feature_row.copy()
    for col in features:
        if col not in X.columns:
            X[col] = np.nan
    X = X[features]

    prob = float(pipeline.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    level, icon = _risk_level_from_prob(prob)

    return {
        "ok": True,
        "risk_probability": prob,
        "risk_score": round(prob * 100, 1),
        "prediction": pred,
        "threshold": threshold,
        "risk_level": level,
        "icon": icon,
        "cell_id": feature_row["cell_id"].iloc[0] if "cell_id" in feature_row.columns else None,
        "feature_end_date": end_date.isoformat(),
    }


def run_onur_inference(
    model_mag: Any,
    model_days: Any,
    metadata: dict,
    events: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    scoped = filter_events_by_date(events, start_date, end_date)
    nearby = filter_near(scoped, lat, lon, radius_km)
    if nearby.empty:
        nearby = filter_near(scoped, lat, lon, radius_km * 2)
    if nearby.empty:
        return {"ok": False, "error": "Seçilen tarih aralığında yakın deprem kaydı yok."}

    feature_list = metadata.get("features", [])
    latest = nearby.sort_values("time").tail(1).iloc[0]
    row: dict[str, Any] = {f: 0.0 for f in feature_list}

    mapping = {
        "magnitude": latest.get("magnitude"),
        "longitude": latest.get("longitude"),
        "latitude": latest.get("latitude"),
        "depth": latest.get("depth_km"),
        "temperature": latest.get("temperature"),
        "humidity": latest.get("humidity"),
        "pressure": latest.get("pressure"),
        "moon_phase": latest.get("moon_phase"),
        "sunspot_number": latest.get("sunspot_number"),
        "solar_flux": latest.get("solar_flux_f107"),
    }
    for k, v in mapping.items():
        if k in row and pd.notna(v):
            row[k] = float(v)

    t = pd.to_datetime(latest.get("time"), errors="coerce")
    if pd.notna(t):
        for key, val in [
            ("year", t.year),
            ("month", t.month),
            ("day", t.day),
            ("day_of_week", t.dayofweek),
            ("season", t.month % 12 // 3 + 1),
        ]:
            if key in row:
                row[key] = val

    window = nearby.sort_values("time")
    if "rolling_mean_magnitude_7d" in row:
        row["rolling_mean_magnitude_7d"] = float(window["magnitude"].mean())
    if "event_count_7d" in row:
        row["event_count_7d"] = float(len(window))

    X = pd.DataFrame([row])[feature_list].astype(float)
    pred_mag = float(model_mag.predict(X)[0])
    pred_days = float(model_days.predict(X)[0])

    trees_mag = np.array([tree.predict(X)[0] for tree in model_mag.estimators_])
    trees_days = np.array([tree.predict(X)[0] for tree in model_days.estimators_])
    conf_mag = 1 / (1 + trees_mag.std() / max(pred_mag, 1e-6))
    conf_days = 1 / (1 + trees_days.std() / max(pred_days, 1e-6))
    confidence = float((conf_mag + conf_days) / 2)

    return {
        "ok": True,
        "pred_max_magnitude_7d": pred_mag,
        "pred_days_to_major": pred_days,
        "confidence": confidence,
        "reference_event_time": str(t) if pd.notna(t) else None,
        "n_nearby_events": len(nearby),
        "date_range": f"{start_date.isoformat()} — {end_date.isoformat()}",
    }


def compute_combined_risk_score(
    rabia: dict,
    onur: dict,
    province_agg: dict | None,
) -> float | None:
    parts = []
    if rabia.get("ok"):
        parts.append(rabia["risk_score"])
    if onur.get("ok"):
        mag = onur.get("pred_max_magnitude_7d", 0)
        days = onur.get("pred_days_to_major", 30)
        conf = onur.get("confidence", 0.5)
        norm_mag = min(mag / 7.0, 1.0)
        norm_days = 1.0 - min(days / 30.0, 1.0)
        onur_score = (0.55 * norm_mag + 0.35 * norm_days + 0.10 * conf) * 100
        parts.append(onur_score)
    if province_agg and not province_agg.get("empty"):
        parts.append(province_agg["mean_probability"] * 100)
    if not parts:
        return None
    return float(np.mean(parts))


def run_live_inference(
    province: str | None,
    lat: float,
    lon: float,
    radius_km: float = DEFAULT_RADIUS_KM,
    date_start: date | None = None,
    date_end: date | None = None,
) -> dict[str, Any]:
    """Tüm modelleri çalıştırır; tarih aralığı ve yarıçap skorları etkiler."""
    if date_end is None:
        date_end = date.today()
    if date_start is None:
        date_start = date_end - timedelta(days=30)

    events = load_raw_events()
    pred_df = load_predictions()
    rabia_bundle = load_rabia_bundle()
    model_mag, model_days = load_onur_models()
    metadata = load_onur_metadata()

    rabia = run_rabia_inference(rabia_bundle, events, lat, lon, end_date=date_end)
    onur = run_onur_inference(
        model_mag, model_days, metadata, events, lat, lon, radius_km, date_start, date_end
    )
    province_agg = aggregate_predictions_near(pred_df, lat, lon, radius_km, date_start, date_end)

    combined = compute_combined_risk_score(rabia, onur, province_agg)

    return {
        "province": province,
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "date_start": date_start.isoformat(),
        "date_end": date_end.isoformat(),
        "rabia": rabia,
        "onur": onur,
        "province_aggregate": province_agg,
        "combined_risk_score": combined,
    }
