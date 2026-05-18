"""Gerçekçi mock veri üreticileri — entegrasyon öncesi demo için."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepfault.config import FORECAST_HORIZON_DAYS, MAG_THRESHOLD, RANDOM_SEED, TURKEY_BOUNDS

RNG = np.random.default_rng(RANDOM_SEED)


def _grid_cells(n_lat: int = 12, n_lon: int = 14) -> pd.DataFrame:
    lats = np.linspace(TURKEY_BOUNDS["lat_min"], TURKEY_BOUNDS["lat_max"], n_lat)
    lons = np.linspace(TURKEY_BOUNDS["lon_min"], TURKEY_BOUNDS["lon_max"], n_lon)
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            rows.append(
                {
                    "cell_id": f"G{i:02d}_{j:02d}",
                    "lat": lat,
                    "lon": lon,
                    "grid_i": i,
                    "grid_j": j,
                }
            )
    return pd.DataFrame(rows)


def mock_data_generator(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Tüm demo veri kümelerini tek çağrıda üretir."""
    rng = np.random.default_rng(seed)
    n_days = 365
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    events = []
    for d in dates:
        n_evt = rng.poisson(2.5)
        for _ in range(n_evt):
            events.append(
                {
                    "time": d + pd.Timedelta(hours=int(rng.integers(0, 24))),
                    "magnitude": float(np.clip(rng.exponential(1.2) + 2.0, 2.0, 7.5)),
                    "depth_km": float(rng.uniform(5, 35)),
                    "latitude": float(rng.uniform(TURKEY_BOUNDS["lat_min"], TURKEY_BOUNDS["lat_max"])),
                    "longitude": float(rng.uniform(TURKEY_BOUNDS["lon_min"], TURKEY_BOUNDS["lon_max"])),
                    "moon_phase": float(rng.uniform(0, 100)),
                    "sunspot_number": float(rng.integers(0, 180)),
                    "solar_flux_f107": float(rng.uniform(65, 220)),
                    "temperature": float(rng.normal(15, 8)),
                    "humidity": float(rng.uniform(20, 95)),
                    "pressure": float(rng.normal(1013, 12)),
                }
            )
    eq_df = pd.DataFrame(events).sort_values("time").reset_index(drop=True)

    astro_df = pd.DataFrame(
        {
            "date": dates,
            "moon_phase": rng.uniform(0, 100, n_days),
            "sunspot_number": rng.integers(0, 200, n_days),
            "solar_flux_f107": rng.uniform(70, 230, n_days),
        }
    )

    weather_df = pd.DataFrame(
        {
            "date": dates,
            "temperature": rng.normal(14, 9, n_days),
            "humidity": rng.uniform(25, 98, n_days),
            "pressure": rng.normal(1012, 14, n_days),
        }
    )

    cells = _grid_cells()
    cells["risk_probability"] = rng.beta(1.5, 6, len(cells))
    cells["risk_score"] = (cells["risk_probability"] * 100).round(1)
    cells["density_increase"] = rng.uniform(-0.2, 0.85, len(cells)).round(3)
    cells["b_value"] = rng.uniform(0.75, 1.35, len(cells)).round(3)
    cells["anomaly_flag"] = (cells["risk_probability"] > 0.55).astype(int)
    cells["energy_accumulation"] = rng.uniform(0, 1, len(cells)).round(3)
    cells["target_mock"] = (cells["risk_probability"] > 0.62).astype(int)

    ts_features = pd.DataFrame(
        {
            "date": dates[-90:],
            "rolling_mag_mean_7d": rng.uniform(2.5, 4.2, 90),
            "rolling_count_7d": rng.integers(5, 45, 90),
            "volatility_14d": rng.uniform(0.1, 1.2, 90),
            "micro_activity_spike": rng.choice([0, 1], 90, p=[0.85, 0.15]),
            "b_value_series": rng.uniform(0.7, 1.4, 90),
        }
    )

    return {
        "earthquakes": eq_df,
        "astro": astro_df,
        "weather": weather_df,
        "grid_cells": cells,
        "time_series_features": ts_features,
        "meta": pd.DataFrame(
            [
                {
                    "forecast_horizon_days": FORECAST_HORIZON_DAYS,
                    "mag_threshold": MAG_THRESHOLD,
                    "model": "XGBoost (mock)",
                    "validation": "Walk-forward (mock)",
                }
            ]
        ),
    }


def compute_b_value_mock(magnitudes: np.ndarray, mc: float = 2.5) -> float:
    """Gutenberg-Richter b-değeri (basitleştirilmiş mock)."""
    mags = magnitudes[magnitudes >= mc]
    if len(mags) < 5:
        return 1.0
    mean_mag = mags.mean()
    return float(np.clip(1 / (0.4343 * (mean_mag - mc + 1e-6)), 0.5, 2.0))


def mock_inference_summary(province: str | None, lat: float, lon: float) -> dict:
    """Anlık çıkarım için özet model çıktısı."""
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 100)) % 2**31)
    prob = float(rng.beta(2, 5))
    return {
        "risk_probability": round(prob, 4),
        "risk_score": round(prob * 100, 1),
        "density_increase_pct": round(rng.uniform(5, 65), 1),
        "anomaly_detected": prob > 0.45,
        "energy_zone": prob > 0.5,
        "b_value": round(rng.uniform(0.8, 1.25), 3),
        "volatility_index": round(rng.uniform(0.2, 0.95), 3),
        "province": province or "Özel Koordinat",
        "latitude": lat,
        "longitude": lon,
        "horizon_days": FORECAST_HORIZON_DAYS,
        "interpretation": (
            f"Önümüzdeki {FORECAST_HORIZON_DAYS} gün içinde bu hücrede M≥{MAG_THRESHOLD} "
            f"olasılığı istatistik olarak {'yükselmiş' if prob > 0.4 else 'düşük'} görünüyor."
        ),
    }
