"""Tek hücre için Rabia modeli özellik üretimi (sızıntısız, geçmişe dayalı)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepfault.geo import cell_id_from_coords
from deepfault.paths import LAT_BIN, LON_BIN, STRONG_MAG


def _assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat_bin"] = (np.round(out["latitude"] / LAT_BIN) * LAT_BIN).astype(float)
    out["lon_bin"] = (np.round(out["longitude"] / LON_BIN) * LON_BIN).astype(float)
    out["cell_id"] = out["lat_bin"].map(lambda x: f"{x:.2f}") + "_" + out["lon_bin"].map(lambda x: f"{x:.2f}")
    return out


def _days_since_binary(x: pd.Series) -> pd.Series:
    arr = x.to_numpy(dtype=np.int8)
    out = np.full(len(arr), np.nan, dtype=float)
    last = None
    for i, val in enumerate(arr):
        if last is not None:
            out[i] = i - last
        if val == 1:
            last = i
    return pd.Series(out, index=x.index)


def build_feature_row_for_location(
    events: pd.DataFrame,
    lat: float,
    lon: float,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Hedef koordinat için son günün özellik satırını üretir (end_date'e kadar)."""
    if events.empty:
        return None

    if end_date is not None:
        events = events[events["time"] <= end_date].copy()
        if events.empty:
            return None

    target_cell = cell_id_from_coords(lat, lon)
    ev = _assign_bins(events)
    ev = ev[ev["cell_id"] == target_cell].copy()
    if ev.empty:
        # Komşu hücrelere genişlet
        lat_bin = round(lat / LAT_BIN) * LAT_BIN
        lon_bin = round(lon / LON_BIN) * LON_BIN
        ev = _assign_bins(events)
        ev = ev[
            (np.abs(ev["lat_bin"] - lat_bin) <= LAT_BIN)
            & (np.abs(ev["lon_bin"] - lon_bin) <= LON_BIN)
        ].copy()
        if ev.empty:
            return None
        ev["cell_id"] = target_cell

    ev["event_date"] = pd.to_datetime(ev["time"], errors="coerce").dt.normalize()
    ev = ev.dropna(subset=["event_date"])

    end = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    start = max(ev["event_date"].min(), end - pd.Timedelta(days=400))

    daily = (
        ev.groupby("event_date")
        .agg(
            event_count=("magnitude", "count"),
            strong_event_count=("magnitude", lambda s: int((s >= STRONG_MAG).sum())),
            max_magnitude=("magnitude", "max"),
            mean_magnitude=("magnitude", "mean"),
            std_magnitude=("magnitude", "std"),
            mean_depth_km=("depth_km", "mean"),
            min_depth_km=("depth_km", "min"),
            pressure=("pressure", "median"),
            temperature=("temperature", "median"),
            moon_phase=("moon_phase", "median"),
            sunspot_number=("sunspot_number", "median"),
            solar_flux_f107=("solar_flux_f107", "median"),
            state=("state", lambda s: s.mode().iloc[0] if len(s) else "unknown"),
            country=("country", lambda s: s.mode().iloc[0] if len(s) else "Turkey"),
            weather_group=("weather_desc", lambda s: s.mode().iloc[0] if len(s) else "no_event"),
        )
        .reset_index()
        .rename(columns={"event_date": "date"})
    )

    all_dates = pd.date_range(start, end, freq="D")
    panel = pd.DataFrame({"date": all_dates})
    panel = panel.merge(daily, on="date", how="left")
    panel["cell_id"] = target_cell
    panel["latitude"] = lat
    panel["longitude"] = lon
    panel["lat_bin"] = round(lat / LAT_BIN) * LAT_BIN
    panel["lon_bin"] = round(lon / LON_BIN) * LON_BIN

    for col in ["event_count", "strong_event_count"]:
        panel[col] = panel[col].fillna(0).astype(int)
    panel["has_event_today"] = (panel["event_count"] > 0).astype(int)
    panel["has_strong_event_today"] = (panel["strong_event_count"] > 0).astype(int)

    for col in ["state", "country", "weather_group"]:
        panel[col] = panel[col].fillna(panel[col].mode().iloc[0] if panel[col].notna().any() else "unknown")

    exog = ["pressure", "temperature", "moon_phase", "sunspot_number", "solar_flux_f107"]
    for col in exog:
        if col in panel.columns:
            panel[col] = panel[col].ffill().bfill().fillna(panel[col].median())

    panel = panel.sort_values("date").reset_index(drop=True).copy()
    gb = panel.groupby("cell_id", group_keys=False)

    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["dayofyear"] = panel["date"].dt.dayofyear
    panel["dayofweek"] = panel["date"].dt.dayofweek
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
    panel["doy_sin"] = np.sin(2 * np.pi * panel["dayofyear"] / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * panel["dayofyear"] / 365.25)

    panel["past_event"] = gb["event_count"].shift(1)
    panel["past_strong"] = gb["strong_event_count"].shift(1)
    panel["past_max_mag"] = gb["max_magnitude"].shift(1)
    panel["past_mean_mag"] = gb["mean_magnitude"].shift(1)
    panel["past_depth"] = gb["mean_depth_km"].shift(1)
    panel["temp_energy"] = (1.5 * panel["max_magnitude"].fillna(0) + 4.8).where(panel["has_event_today"] == 1, 0)
    panel["past_energy"] = gb["temp_energy"].shift(1)
    panel.drop(columns=["temp_energy"], inplace=True)

    for w in [1, 3, 7, 14, 30, 90, 180, 365]:
        panel[f"event_count_past_{w}d"] = gb["past_event"].transform(lambda x, w=w: x.rolling(w, min_periods=1).sum())
        panel[f"strong_count_past_{w}d"] = gb["past_strong"].transform(lambda x, w=w: x.rolling(w, min_periods=1).sum())
        panel[f"energy_log_sum_past_{w}d"] = gb["past_energy"].transform(lambda x, w=w: x.rolling(w, min_periods=1).sum())

    for w in [7, 30, 90, 180]:
        panel[f"max_mag_past_{w}d"] = gb["past_max_mag"].transform(lambda x, w=w: x.rolling(w, min_periods=1).max())
        panel[f"mean_mag_past_{w}d"] = gb["past_mean_mag"].transform(lambda x, w=w: x.rolling(w, min_periods=2).mean())
        panel[f"std_mag_past_{w}d"] = gb["past_mean_mag"].transform(lambda x, w=w: x.rolling(w, min_periods=3).std())
        panel[f"depth_mean_past_{w}d"] = gb["past_depth"].transform(lambda x, w=w: x.rolling(w, min_periods=2).mean())

    for lag in [1, 2, 3, 7, 14, 30]:
        panel[f"event_count_lag_{lag}d"] = gb["event_count"].shift(lag)
        panel[f"strong_count_lag_{lag}d"] = gb["strong_event_count"].shift(lag)
        panel[f"max_mag_lag_{lag}d"] = gb["max_magnitude"].shift(lag)

    panel["days_since_last_event"] = gb["event_count"].transform(
        lambda x: _days_since_binary((x > 0).astype(int))
    )
    panel["days_since_last_strong_event"] = gb["strong_event_count"].transform(
        lambda x: _days_since_binary((x > 0).astype(int))
    )

    panel["event_acceleration_7_vs_30"] = panel["event_count_past_7d"] - (panel["event_count_past_30d"] / 30 * 7)
    panel["strong_acceleration_7_vs_90"] = panel["strong_count_past_7d"] - (panel["strong_count_past_90d"] / 90 * 7)
    panel["energy_acceleration_7_vs_30"] = panel["energy_log_sum_past_7d"] - (panel["energy_log_sum_past_30d"] / 30 * 7)

    if "pressure" in panel.columns:
        panel["pressure_change_1d"] = gb["pressure"].diff(1)
        panel["pressure_change_3d"] = gb["pressure"].diff(3)
        past_max_7 = gb["pressure"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
        panel["pressure_drop_7d"] = past_max_7 - panel["pressure"]

    if "temperature" in panel.columns:
        doy_med = panel.groupby("dayofyear")["temperature"].transform("median")
        panel["thermal_anomaly"] = panel["temperature"] - doy_med
        panel["temperature_change_1d"] = gb["temperature"].diff(1)

    if "moon_phase" in panel.columns:
        mp = panel["moon_phase"].astype(float)
        denom = 100.0 if mp.max(skipna=True) > 30 else 29.530588
        panel["moon_sin"] = np.sin(2 * np.pi * mp / denom)
        panel["moon_cos"] = np.cos(2 * np.pi * mp / denom)
        panel["tidal_stress_proxy"] = panel["moon_cos"]

    if "solar_flux_f107" in panel.columns:
        panel["solar_flux_7d_mean"] = gb["solar_flux_f107"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).mean()
        )
        panel["solar_flux_30d_mean"] = gb["solar_flux_f107"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=1).mean()
        )
        panel["solar_flux_change_1d"] = gb["solar_flux_f107"].diff(1)

    lat_v = panel["latitude"].astype(float)
    lon_v = panel["longitude"].astype(float)
    kaf_line = 40.7 - 0.025 * (lon_v - 30)
    daf_line = 38.2 - 0.18 * (lon_v - 36)
    baf_line = 38.6 + 0.05 * (lon_v - 28)
    km = 111.0
    panel["fault_proximity_proxy_km"] = np.minimum.reduce(
        [np.abs(lat_v - kaf_line) * km, np.abs(lat_v - daf_line) * km, np.abs(lat_v - baf_line) * km]
    )

    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.tail(1).reset_index(drop=True).copy()
