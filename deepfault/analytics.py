"""Gerçek veri analitiği — mock yok."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from deepfault.geo import filter_near, haversine_km


def filter_predictions_by_date(
    pred_df: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    if pred_df.empty or "date" not in pred_df.columns:
        return pred_df
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (pred_df["date"] >= start_ts) & (pred_df["date"] <= end_ts)
    return pred_df.loc[mask].copy()


def filter_events_by_date(
    events: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    if events.empty or "time" not in events.columns:
        return events
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return events.loc[(events["time"] >= start_ts) & (events["time"] <= end_ts)].copy()


def aggregate_predictions_near(
    pred_df: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
    start: date,
    end: date,
) -> dict:
    """Tarih aralığı ve yarıçapa göre grid risk agregasyonu."""
    scoped = filter_predictions_by_date(pred_df, start, end)
    if scoped.empty:
        return {"empty": True}

    dist = haversine_km(lat, lon, scoped["latitude"].to_numpy(), scoped["longitude"].to_numpy())
    scoped = scoped.assign(distance_km=dist)
    local = scoped[scoped["distance_km"] <= radius_km]
    if local.empty:
        local = scoped.nsmallest(max(25, int(25 * radius_km / 75)), "distance_km")

    if local.empty:
        return {"empty": True}

    return {
        "empty": False,
        "mean_probability": float(local["risk_score"].mean()),
        "max_probability": float(local["risk_score"].max()),
        "median_probability": float(local["risk_score"].median()),
        "std_probability": float(local["risk_score"].std()) if len(local) > 1 else 0.0,
        "n_points": int(len(local)),
        "n_cells": int(local["cell_id"].nunique()) if "cell_id" in local.columns else len(local),
        "date_start": start.isoformat(),
        "date_end": end.isoformat(),
        "positive_predictions": int((local.get("prediction", 0) == 1).sum()) if "prediction" in local.columns else None,
        "data": local,
    }


def regional_risk_timeseries(
    pred_df: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    scoped = filter_predictions_by_date(pred_df, start, end)
    if scoped.empty:
        return pd.DataFrame()

    dist = haversine_km(lat, lon, scoped["latitude"].to_numpy(), scoped["longitude"].to_numpy())
    local = scoped.assign(distance_km=dist)
    local = local[local["distance_km"] <= radius_km]
    if local.empty:
        local = scoped.assign(distance_km=dist).nsmallest(25, "distance_km")

    daily = (
        local.groupby("date", as_index=False)
        .agg(mean_risk=("risk_score", "mean"), max_risk=("risk_score", "max"), n_cells=("cell_id", "nunique"))
        .sort_values("date")
    )
    return daily


def map_cells_in_range(
    pred_df: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Tarih aralığında hücre bazlı ortalama risk."""
    scoped = filter_predictions_by_date(pred_df, start, end)
    if scoped.empty:
        return scoped

    agg = (
        scoped.groupby(["cell_id", "latitude", "longitude"], as_index=False)
        .agg(
            risk_score=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            n_days=("date", "nunique"),
        )
    )
    return agg


def seismic_stats_near(
    events: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
    start: date,
    end: date,
) -> dict:
    scoped = filter_events_by_date(events, start, end)
    local = filter_near(scoped, lat, lon, radius_km)
    if local.empty:
        local = filter_near(scoped, lat, lon, radius_km * 2)

    if local.empty:
        return {"empty": True}

    mags = local["magnitude"]
    return {
        "empty": False,
        "event_count": len(local),
        "mean_magnitude": float(mags.mean()),
        "max_magnitude": float(mags.max()),
        "strong_events": int((mags >= 4.0).sum()),
        "mean_depth_km": float(local["depth_km"].mean()) if "depth_km" in local.columns else None,
    }
