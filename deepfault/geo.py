"""Coğrafi yardımcılar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepfault.paths import LAT_BIN, LON_BIN


def cell_id_from_coords(lat: float, lon: float) -> str:
    lat_bin = round(lat / LAT_BIN) * LAT_BIN
    lon_bin = round(lon / LON_BIN) * LON_BIN
    return f"{lat_bin:.2f}_{lon_bin:.2f}"


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def filter_near(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    if df.empty:
        return df
    dist = haversine_km(lat, lon, df[lat_col].to_numpy(), df[lon_col].to_numpy())
    return df.loc[dist <= radius_km].copy()
