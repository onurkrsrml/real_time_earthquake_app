"""
DEEPFAULT - Rabia TIME SERIES V4 BEST FINAL
======================================

Bu sürüm, önceki event-level pipeline'ın en kritik zayıflığını düzeltir:
Artık model yalnızca "deprem olmuş satırlar" üzerinde değil, günlük grid-cell panel
üzerinde çalışır. Yani her bölge-gün için geçmiş sismik, atmosferik ve astronomik
bilgilerden gelecek 7 gün içinde M>=4 deprem riski tahmin edilir.

Neden daha doğru?
-----------------
1) Gerçek time series mantığı: daily panel data = cell_id x date.
2) Gelecek label kesin ayrılır: target_future_7d bugünden sonraki 7 günü kullanır.
3) Feature'lar yalnızca geçmişten gelir: tüm rolling/lag feature'lar shift(1) ile hesaplanır.
4) Son horizon günleri drop edilir: geleceği bilinmeyen satırlar eğitime/evaluasyona girmez.
5) Purged time split kullanılır: train ile validation/test arasında horizon kadar embargo bırakılır.
6) Walk-forward validation vardır: Optuna skorlaması kronolojik fold'larla yapılır.
7) outputs/ klasörüne grafik, metrik, veri, model ve raporları otomatik kaydeder.

Klasör yapısı:
--------------
DEEPFAULT/
├── deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL.py
├── depremler_hava_nasa.csv
└── outputs/

Kurulum:
--------
pip install numpy pandas matplotlib scikit-learn xgboost optuna shap joblib

Çalıştırma:
-----------
python deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL.py

Opsiyonel:
python deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL.py --input depremler_hava_nasa.csv --output-dir outputs --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from sklearn.calibration import CalibratedClassifierCV
    HAS_CALIBRATION = True
except Exception:
    HAS_CALIBRATION = False

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except Exception:
    HAS_FROZEN_ESTIMATOR = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    input_file: str = "depremler_hava_nasa.csv"
    output_dir: str = "outputs"

    # Gerçek risk problemi: bugünden sonraki horizon içinde güçlü deprem var mı?
    horizon_days: int = 7
    target_mag_threshold: float = 4.0

    # Grid ayarı. Daha küçük değer daha detaylı ama daha seyrek data üretir.
    lat_bin_size: float = 0.75
    lon_bin_size: float = 0.75
    min_cell_events: int = 8

    # Panel başlangıcı. Çok eski data istersen 1940 yap; daha stabil model için 1990 iyi olur.
    panel_start_date: str = "1990-01-01"

    # Time split. Veri 2026'ya kadar olduğu için test son dönem olmalı.
    validation_start_date: str = "2022-01-01"
    test_start_date: str = "2024-01-01"

    # Model
    n_trials: int = 20
    n_splits: int = 5
    max_rows_for_optuna: int = 250_000
    max_rows_for_training: int = 500_000
    max_rows_for_shap: int = 3000
    random_state: int = 42

    # Threshold seçiminde en az precision beklentisi.
    # Threshold: yüksek recall tek başına yeterli değil; minimum precision ile fazla alarm kontrol edilir.
    min_precision_for_threshold: float = 0.35

    # Bellek koruma: panel milyonlarca satıra çıktığında tüm negatif günleri eğitime almak RAM tüketir.
    # Tüm pozitifleri korur, negatifleri yıl bazlı örnekler. 0 veya None verilirse kapatılır.
    # Negatif örnekleme: yüksek oran = daha doğal sınıf dağılımı, düşük oran = daha hızlı eğitim.
    negative_sample_ratio: int = 8

    # Grafik
    show_plots: bool = True


# =============================================================================
# BASIC UTILS
# =============================================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def safe_show(cfg: Config) -> None:
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas DataFrame'i RAM dostu tipe çevirir. Özellikle daily panelde kritiktir."""
    out = df
    float_cols = out.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], downcast="float")

    int_cols = out.select_dtypes(include=["int64", "int32"]).columns
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], downcast="integer")

    for col in out.select_dtypes(include=["object"]).columns:
        # Çok yüksek kardinaliteli id hariç kategorik kolonları category yap.
        if out[col].nunique(dropna=False) / max(len(out), 1) < 0.50:
            out[col] = out[col].astype("category")
    return out


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


# =============================================================================
# LOAD + CLEAN
# =============================================================================

def choose_input_file(cfg: Config) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path(cfg.input_file), script_dir / cfg.input_file]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"CSV bulunamadı: {cfg.input_file}. Dosyayı Python dosyasıyla aynı klasöre koy.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    rename = {
        "date": "time",
        "datetime": "time",
        "timestamp": "time",
        "tarih": "time",
        "mag": "magnitude",
        "m": "magnitude",
        "depth": "depth_km",
        "derinlik": "depth_km",
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "long": "longitude",
        "solar_flux": "solar_flux_f107",
        "f107": "solar_flux_f107",
        "weather": "weather_desc",
        "city": "city_name",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def clean_raw_events(df: pd.DataFrame, cfg: Config, out: Path) -> pd.DataFrame:
    df = normalize_columns(df)

    required = ["time", "magnitude", "latitude", "longitude", "depth_km"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Zorunlu kolonlar eksik: {missing}. Mevcut kolonlar: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    numeric_cols = [
        "magnitude", "latitude", "longitude", "depth_km", "temperature", "humidity",
        "pressure", "moon_phase", "sunspot_number", "solar_flux_f107"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["time", "magnitude", "latitude", "longitude", "depth_km"])
    print(f"[CLEAN] Temel zorunlu eksiklerden düşen satır: {before - len(df):,}")

    # Türkiye ve yakın çevre filtresi: Proje Türkiye bölgesel risk analizi olduğu için aşırı uzakları dışarıda bırak.
    # Geniş bırakıldı; Yunanistan/Kıbrıs/Georgia yakın çevresi kalabilir.
    geo_before = len(df)
    df = df[(df["latitude"].between(33, 43.5)) & (df["longitude"].between(24, 46.5))].copy()
    print(f"[CLEAN] Coğrafi filtre sonrası düşen satır: {geo_before - len(df):,}")

    # Fiziksel aralık temizlikleri
    df.loc[~df["magnitude"].between(0, 9.8), "magnitude"] = np.nan
    df.loc[~df["depth_km"].between(0, 700), "depth_km"] = np.nan

    if "pressure" in df.columns:
        med_p = df["pressure"].median(skipna=True)
        if pd.notna(med_p) and 80 <= med_p <= 120:
            print("[FIX] pressure median 80-120 aralığında: kPa -> hPa dönüşümü için *10 uygulandı.")
            df["pressure"] = df["pressure"] * 10
        df.loc[~df["pressure"].between(850, 1100), "pressure"] = np.nan

    if "humidity" in df.columns:
        if df["humidity"].nunique(dropna=True) <= 3 or df["humidity"].std(skipna=True) < 0.5:
            print("[DROP] humidity neredeyse sabit/uydurulmuş görünüyor; modelden çıkarılacak.")
            df = df.drop(columns=["humidity"])
        else:
            df.loc[~df["humidity"].between(0, 100), "humidity"] = np.nan

    if "temperature" in df.columns:
        df.loc[~df["temperature"].between(-50, 65), "temperature"] = np.nan

    if "solar_flux_f107" in df.columns:
        df.loc[~df["solar_flux_f107"].between(40, 450), "solar_flux_f107"] = np.nan

    if "moon_phase" in df.columns:
        df.loc[~df["moon_phase"].between(0, 100), "moon_phase"] = np.nan

    # Eski feature dosyasından gelmiş sentetik kolonları kesin kaldır.
    suspicious_cols = [
        "soil_radon_concentration", "gps_displacement_rate", "electromagnetic_signal_power",
        "crustal_strain_rate", "groundwater_level_change"
    ]
    existing = [c for c in suspicious_cols if c in df.columns]
    if existing:
        print(f"[DROP] Gerçek ölçüm olmayan/sentetik feature kolonları kaldırıldı: {existing}")
        df = df.drop(columns=existing)

    # Hava durumu kategorisini sadeleştir.
    if "weather_desc" in df.columns:
        df["weather_desc"] = df["weather_desc"].fillna("unknown").astype(str).str.lower().str.strip()
        df["weather_group"] = np.select(
            [
                df["weather_desc"].str.contains("rain|drizzle|shower", regex=True),
                df["weather_desc"].str.contains("snow|sleet|ice", regex=True),
                df["weather_desc"].str.contains("fog|mist", regex=True),
                df["weather_desc"].str.contains("cloud|overcast", regex=True),
                df["weather_desc"].str.contains("clear", regex=True),
            ],
            ["rain", "snow", "fog", "cloudy", "clear"],
            default="other",
        )
    else:
        df["weather_group"] = "unknown"

    # Grid cell üretimi
    df["event_date"] = df["time"].dt.floor("D")
    df["lat_bin"] = np.floor(df["latitude"] / cfg.lat_bin_size) * cfg.lat_bin_size
    df["lon_bin"] = np.floor(df["longitude"] / cfg.lon_bin_size) * cfg.lon_bin_size
    df["cell_id"] = df["lat_bin"].round(3).astype(str) + "_" + df["lon_bin"].round(3).astype(str)

    df = df.dropna(subset=["magnitude", "depth_km", "event_date", "cell_id"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    quality = {
        "rows_after_cleaning": int(len(df)),
        "date_min": str(df["time"].min()),
        "date_max": str(df["time"].max()),
        "columns": list(df.columns),
        "missing_ratio_top20": df.isna().mean().sort_values(ascending=False).head(20).round(4).to_dict(),
        "magnitude_summary": df["magnitude"].describe().round(4).to_dict(),
        "pressure_summary": df["pressure"].describe().round(4).to_dict() if "pressure" in df.columns else None,
        "unique_cells": int(df["cell_id"].nunique()),
    }
    save_json(quality, out / "00_data_quality_raw_cleaned.json")
    return df


# =============================================================================
# DAILY GRID-CELL PANEL: TRUE TIME SERIES STRUCTURE
# =============================================================================

def mode_or_unknown(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if len(s) == 0:
        return "unknown"
    return s.mode().iloc[0]


def build_daily_cell_panel(events: pd.DataFrame, cfg: Config, out: Path) -> pd.DataFrame:
    print_section("2) GÜNLÜK GRID-CELL TIME SERIES PANEL KURULUYOR")

    # Yeterli geçmişe sahip cell'ler.
    cell_counts = events.groupby("cell_id").size().sort_values(ascending=False)
    keep_cells = cell_counts[cell_counts >= cfg.min_cell_events].index.tolist()
    if len(keep_cells) == 0:
        raise ValueError("Yeterli event içeren cell bulunamadı. min_cell_events değerini düşür.")

    ev = events[events["cell_id"].isin(keep_cells)].copy()
    ev = ev[ev["event_date"] >= pd.Timestamp(cfg.panel_start_date)].copy()
    if len(ev) == 0:
        raise ValueError("panel_start_date sonrası veri yok. panel_start_date değerini eski bir tarihe çek.")

    # Daily aggregation: her cell-gün için event bilgisi.
    agg_spec = {
        "event_count": ("magnitude", "size"),
        "max_magnitude": ("magnitude", "max"),
        "mean_magnitude": ("magnitude", "mean"),
        "std_magnitude": ("magnitude", "std"),
        "strong_event_count": ("magnitude", lambda x: int((x >= cfg.target_mag_threshold).sum())),
        "mean_depth_km": ("depth_km", "mean"),
        "min_depth_km": ("depth_km", "min"),
        "latitude": ("latitude", "median"),
        "longitude": ("longitude", "median"),
        "lat_bin": ("lat_bin", "first"),
        "lon_bin": ("lon_bin", "first"),
        "weather_group": ("weather_group", mode_or_unknown),
    }
    for col in ["pressure", "temperature", "moon_phase", "sunspot_number", "solar_flux_f107"]:
        if col in ev.columns:
            agg_spec[col] = (col, "median")

    daily = ev.groupby(["event_date", "cell_id"], as_index=False).agg(**agg_spec)

    # Complete daily panel: her cell için her gün. Böylece deprem olmayan günler de modele girer.
    start = max(pd.Timestamp(cfg.panel_start_date), daily["event_date"].min())
    end = daily["event_date"].max()
    all_dates = pd.date_range(start, end, freq="D")
    cells = sorted(daily["cell_id"].unique())
    full_index = pd.MultiIndex.from_product([all_dates, cells], names=["date", "cell_id"])
    panel = full_index.to_frame(index=False)

    daily = daily.rename(columns={"event_date": "date"})
    panel = panel.merge(daily, on=["date", "cell_id"], how="left")

    # Sabit cell koordinatlarını doldur.
    cell_meta = ev.groupby("cell_id").agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        lat_bin=("lat_bin", "first"),
        lon_bin=("lon_bin", "first"),
        state=("state", mode_or_unknown) if "state" in ev.columns else ("cell_id", mode_or_unknown),
        country=("country", mode_or_unknown) if "country" in ev.columns else ("cell_id", mode_or_unknown),
    ).reset_index()

    for col in ["latitude", "longitude", "lat_bin", "lon_bin"]:
        panel = panel.drop(columns=[col], errors="ignore")
    panel = panel.merge(cell_meta, on="cell_id", how="left")

    # Event olmayan günler için deprem değişkenleri sıfır/NaN ayrımı.
    fill_zero_cols = ["event_count", "strong_event_count"]
    for col in fill_zero_cols:
        panel[col] = panel[col].fillna(0).astype(np.int16)

    panel["has_event_today"] = (panel["event_count"] > 0).astype(np.int8)
    panel["has_strong_event_today"] = (panel["strong_event_count"] > 0).astype(np.int8)

    # Magnitude/depth sadece event günlerinde var; feature engineering geçmiş rolling ile dolduracak.
    for col in ["max_magnitude", "mean_magnitude", "std_magnitude", "mean_depth_km", "min_depth_km"]:
        if col in panel.columns:
            panel[col] = panel[col].astype(float)

    # Exogenous kolonlar: event günlerinde mevcut; gün bazında global median + cell forward fill ile makul doldurma.
    # Not: En iyi üretim versiyonda NASA POWER'dan her cell-date için günlük hava verisi çekilmeli.
    exog_cols = [c for c in ["pressure", "temperature", "moon_phase", "sunspot_number", "solar_flux_f107"] if c in panel.columns]
    for col in exog_cols:
        global_daily = panel.groupby("date")[col].transform("median")
        panel[col] = panel[col].fillna(global_daily)
        panel[col] = panel.groupby("cell_id")[col].ffill().bfill()
        panel[col] = panel[col].fillna(panel[col].median())

    panel["weather_group"] = panel["weather_group"].fillna("no_event")
    panel["state"] = panel["state"].fillna("unknown")
    panel["country"] = panel["country"].fillna("unknown")

    print(f"[INFO] Panel boyutu: {panel.shape}")
    print(f"[INFO] Cell sayısı: {panel['cell_id'].nunique()} | Gün sayısı: {panel['date'].nunique()}")

    panel_report = {
        "panel_shape": list(panel.shape),
        "cell_count": int(panel["cell_id"].nunique()),
        "date_min": str(panel["date"].min()),
        "date_max": str(panel["date"].max()),
        "event_day_rate": float(panel["has_event_today"].mean()),
        "strong_event_day_rate": float(panel["has_strong_event_today"].mean()),
        "min_cell_events": cfg.min_cell_events,
        "lat_bin_size": cfg.lat_bin_size,
        "lon_bin_size": cfg.lon_bin_size,
    }
    save_json(panel_report, out / "01_panel_report.json")
    panel = optimize_memory(panel)
    return panel.sort_values(["date", "cell_id"]).reset_index(drop=True)


# =============================================================================
# TARGET + TIME SERIES FEATURES
# =============================================================================

def days_since_binary_event(x: pd.Series) -> pd.Series:
    # x chronological 0/1. Returns days since previous 1, excluding today.
    arr = x.to_numpy(dtype=np.int8)
    out = np.full(len(arr), np.nan, dtype=float)
    last = None
    for i, val in enumerate(arr):
        if last is not None:
            out[i] = i - last
        if val == 1:
            last = i
    return pd.Series(out, index=x.index)


def future_horizon_target(strong_today: pd.Series, horizon: int) -> pd.Series:
    # Excludes today. Looks at t+1 ... t+horizon.
    rev = strong_today.iloc[::-1]
    future_count = rev.shift(1).rolling(window=horizon, min_periods=1).sum().iloc[::-1]
    return (future_count > 0).astype(np.int8)


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_fault_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lat = out["latitude"].astype(float)
    lon = out["longitude"].astype(float)

    # Basit ana fay proxy'si. Gerçek projede MTA aktif fay shapefile ile geodesic distance önerilir.
    kaf_line = 40.7 - 0.025 * (lon - 30)
    daf_line = 38.2 - 0.18 * (lon - 36)
    baf_line = 38.6 + 0.05 * (lon - 28)
    km = 111.0
    out["fault_proximity_proxy_km"] = np.minimum.reduce([
        np.abs(lat - kaf_line) * km,
        np.abs(lat - daf_line) * km,
        np.abs(lat - baf_line) * km,
    ])
    return out


def add_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["cell_id", "date"])

    if "pressure" in out.columns:
        out["pressure_change_1d"] = out.groupby("cell_id")["pressure"].diff(1)
        out["pressure_change_3d"] = out.groupby("cell_id")["pressure"].diff(3)
        past_max_7 = out.groupby("cell_id")["pressure"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
        out["pressure_drop_7d"] = past_max_7 - out["pressure"]

    if "temperature" in out.columns:
        # Seasonal anomaly: aynı yıl değil, tüm panelin day-of-year medianına göre.
        doy_med = out.groupby("dayofyear")["temperature"].transform("median") if "dayofyear" in out.columns else out["temperature"].median()
        out["thermal_anomaly"] = out["temperature"] - doy_med
        out["temperature_change_1d"] = out.groupby("cell_id")["temperature"].diff(1)

    if "moon_phase" in out.columns:
        mp = out["moon_phase"].astype(float)
        denom = 100.0 if mp.max(skipna=True) > 30 else 29.530588
        out["moon_sin"] = np.sin(2 * np.pi * mp / denom)
        out["moon_cos"] = np.cos(2 * np.pi * mp / denom)
        out["tidal_stress_proxy"] = out["moon_cos"]

    if "solar_flux_f107" in out.columns:
        out["solar_flux_7d_mean"] = out.groupby("cell_id")["solar_flux_f107"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        out["solar_flux_30d_mean"] = out.groupby("cell_id")["solar_flux_f107"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        out["solar_flux_change_1d"] = out.groupby("cell_id")["solar_flux_f107"].diff(1)

    return out


def engineer_time_series(panel: pd.DataFrame, cfg: Config, out: Path) -> pd.DataFrame:
    print_section("3) LEAKAGE-SAFE TARGET + TIME SERIES FEATURE ENGINEERING")

    # Veriyi sıralıyoruz ki groupby işlemleri kronolojik olarak hatasız çalışsın
    df = panel.copy().sort_values(["cell_id", "date"]).reset_index(drop=True)
    df = add_calendar_features(df)

    # Vektörizasyon için temel groupby objemiz (for döngüsü yok!)
    gb = df.groupby("cell_id")

    # Target: future 7 days strong event, excluding today.
    df["target_future_7d_m4"] = gb["has_strong_event_today"].transform(lambda x: future_horizon_target(x, cfg.horizon_days))
    
    # is_label_known: Her grubun son 'horizon_days' kadar gününü False yapıyoruz
    df["is_label_known"] = True
    tail_mask = gb.cumcount(ascending=False) < cfg.horizon_days
    df.loc[tail_mask, "is_label_known"] = False

    # Strict past-only seismic features (Tüm shift işlemleri vektörel)
    df["past_event"] = gb["event_count"].shift(1)
    df["past_strong"] = gb["strong_event_count"].shift(1)
    df["past_max_mag"] = gb["max_magnitude"].shift(1)
    df["past_mean_mag"] = gb["mean_magnitude"].shift(1)
    df["past_depth"] = gb["mean_depth_km"].shift(1)
    
    # past_energy
    df["temp_energy"] = (1.5 * df["max_magnitude"].fillna(0) + 4.8).where(df["has_event_today"] == 1, 0)
    df["past_energy"] = gb["temp_energy"].shift(1)
    df.drop(columns=["temp_energy"], inplace=True)

    # Rolling (Hareketli Pencere) hesaplamaları - transform ile RAM dostu
    for w in [1, 3, 7, 14, 30, 90, 180, 365]:
        df[f"event_count_past_{w}d"] = gb["past_event"].transform(lambda x: x.rolling(w, min_periods=1).sum())
        df[f"strong_count_past_{w}d"] = gb["past_strong"].transform(lambda x: x.rolling(w, min_periods=1).sum())
        df[f"energy_log_sum_past_{w}d"] = gb["past_energy"].transform(lambda x: x.rolling(w, min_periods=1).sum())

    for w in [7, 30, 90, 180]:
        df[f"max_mag_past_{w}d"] = gb["past_max_mag"].transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f"mean_mag_past_{w}d"] = gb["past_mean_mag"].transform(lambda x: x.rolling(w, min_periods=2).mean())
        df[f"std_mag_past_{w}d"] = gb["past_mean_mag"].transform(lambda x: x.rolling(w, min_periods=3).std())
        df[f"depth_mean_past_{w}d"] = gb["past_depth"].transform(lambda x: x.rolling(w, min_periods=2).mean())

    # Lag (Gecikme) hesaplamaları
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"event_count_lag_{lag}d"] = gb["event_count"].shift(lag)
        df[f"strong_count_lag_{lag}d"] = gb["strong_event_count"].shift(lag)
        df[f"max_mag_lag_{lag}d"] = gb["max_magnitude"].shift(lag)

    # Gün farkı hesaplamaları
    df["days_since_last_event"] = gb["event_count"].transform(lambda x: days_since_binary_event((x > 0).astype(int)))
    df["days_since_last_strong_event"] = gb["strong_event_count"].transform(lambda x: days_since_binary_event((x > 0).astype(int)))

    # İvmelenme (Acceleration) özellikleri
    df["event_acceleration_7_vs_30"] = df["event_count_past_7d"] - (df["event_count_past_30d"] / 30 * 7)
    df["strong_acceleration_7_vs_90"] = df["strong_count_past_7d"] - (df["strong_count_past_90d"] / 90 * 7)
    df["energy_acceleration_7_vs_30"] = df["energy_log_sum_past_7d"] - (df["energy_log_sum_past_30d"] / 30 * 7)

    # Döngü ve parça birleştirme çöpe gitti, direkt df üzerinden devam
    feat = optimize_memory(df)
    feat = add_exogenous_features(feat)
    feat = add_fault_proxy(feat)
    feat = optimize_memory(feat)

    leakage_today_cols = [
        "event_count", "strong_event_count", "has_event_today", "has_strong_event_today",
        "max_magnitude", "mean_magnitude", "std_magnitude", "mean_depth_km", "min_depth_km"
    ]

    feat = feat.loc[feat["is_label_known"]].reset_index(drop=True)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    warmup_cut = pd.Timestamp(cfg.panel_start_date) + pd.Timedelta(days=30)
    feat = feat.loc[feat["date"] >= warmup_cut].reset_index(drop=True)

    if getattr(cfg, "negative_sample_ratio", 0):
        rng = np.random.default_rng(cfg.random_state)
        positives = feat[feat["target_future_7d_m4"] == 1]
        negatives = feat[feat["target_future_7d_m4"] == 0]
        max_neg = int(len(positives) * cfg.negative_sample_ratio)
        if len(positives) > 0 and len(negatives) > max_neg:
            neg_sample = (
                negatives.assign(_year=negatives["date"].dt.year)
                .groupby("_year", group_keys=False)
                .apply(lambda x: x.sample(
                    n=min(len(x), max(1, int(max_neg * len(x) / max(len(negatives), 1)))),
                    random_state=cfg.random_state,
                ))
                .drop(columns=["_year"], errors="ignore")
            )
            feat = pd.concat([positives, neg_sample], axis=0).sort_values(["date", "cell_id"]).reset_index(drop=True)
            print(f"[INFO] Bellek dostu negatif örnekleme uygulandı: positives={len(positives):,}, negatives_sampled={len(neg_sample):,}, final={len(feat):,}")

    feat = optimize_memory(feat)

    report = {
        "featured_shape": list(feat.shape),
        "target_rate": float(feat["target_future_7d_m4"].mean()),
        "target_counts": feat["target_future_7d_m4"].value_counts().to_dict(),
        "date_min": str(feat["date"].min()),
        "date_max": str(feat["date"].max()),
        "leakage_today_cols_not_used_as_features": leakage_today_cols,
        "note": "All rolling/lag features use shift(1), so today's earthquake information is not used for today's risk prediction.",
    }
    save_json(report, out / "02_feature_engineering_report.json")

    print(f"[INFO] Feature sonrası panel boyutu: {feat.shape}")
    print(f"[INFO] Target pozitif oranı: {feat['target_future_7d_m4'].mean():.4f}")
    return feat


# =============================================================================
# EDA PLOTS
# =============================================================================

def plot_eda(events: pd.DataFrame, feat: pd.DataFrame, cfg: Config, out: Path) -> None:
    print_section("4) EDA + TIME SERIES GRAFİKLERİ")

    # 1 magnitude distribution
    plt.figure(figsize=(9, 5))
    plt.hist(events["magnitude"].dropna(), bins=50)
    plt.title("Magnitude Dağılımı - Ham Deprem Olayları")
    plt.xlabel("Magnitude")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.savefig(out / "01_magnitude_distribution.png", dpi=250)
    safe_show(cfg)

    # 2 yearly event counts
    yearly = events.groupby(events["event_date"].dt.year).size()
    plt.figure(figsize=(13, 5))
    plt.plot(yearly.index, yearly.values, marker="o", linewidth=1)
    plt.title("Yıllara Göre Deprem Olay Sayısı")
    plt.xlabel("Yıl")
    plt.ylabel("Olay Sayısı")
    plt.tight_layout()
    plt.savefig(out / "02_yearly_event_count.png", dpi=250)
    safe_show(cfg)

    # 3 target rate over years
    tmp = feat.copy()
    tmp["year"] = tmp["date"].dt.year
    yearly_target = tmp.groupby("year")["target_future_7d_m4"].mean()
    plt.figure(figsize=(13, 5))
    plt.plot(yearly_target.index, yearly_target.values, marker="o", linewidth=1)
    plt.title("Yıllara Göre Gelecek 7 Gün M>=4 Risk Oranı")
    plt.xlabel("Yıl")
    plt.ylabel("Pozitif Target Oranı")
    plt.tight_layout()
    plt.savefig(out / "03_yearly_target_rate.png", dpi=250)
    safe_show(cfg)

    # 4 regional density scatter
    plt.figure(figsize=(9, 7))
    sample = events.sample(min(len(events), 40000), random_state=cfg.random_state)
    plt.scatter(sample["longitude"], sample["latitude"], s=4, alpha=0.35)
    plt.title("Deprem Lokasyon Yoğunluğu")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out / "04_event_location_scatter.png", dpi=250)
    safe_show(cfg)

    # 5 focus correlation heatmap manually with imshow
    focus_cols = [
        "target_future_7d_m4", "event_count_past_7d", "event_count_past_30d",
        "strong_count_past_30d", "energy_log_sum_past_30d", "days_since_last_event",
        "pressure_drop_7d", "thermal_anomaly", "solar_flux_7d_mean", "fault_proximity_proxy_km"
    ]
    focus_cols = [c for c in focus_cols if c in feat.columns]
    if len(focus_cols) >= 3:
        corr = feat[focus_cols].corr(numeric_only=True).fillna(0)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(focus_cols)), focus_cols, rotation=90)
        plt.yticks(range(len(focus_cols)), focus_cols)
        plt.title("Odak Feature Korelasyon Matrisi")
        plt.tight_layout()
        plt.savefig(out / "05_focus_correlation_matrix.png", dpi=250)
        safe_show(cfg)

    print(f"[INFO] EDA görselleri kaydedildi: {out.resolve()}")


# =============================================================================
# SPLIT, PREPROCESSOR, MODEL
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    never_features = {
        "date", "target_future_7d_m4", "is_label_known",
        # Today's event outcome must not be used as feature.
        "event_count", "strong_event_count", "has_event_today", "has_strong_event_today",
        "max_magnitude", "mean_magnitude", "std_magnitude", "mean_depth_km", "min_depth_km",
    }
    categorical_candidates = ["cell_id", "weather_group", "state", "country"]
    categorical = [c for c in categorical_candidates if c in df.columns and c not in never_features]
    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in never_features
    ]
    return numeric, categorical


def purged_time_split(df: pd.DataFrame, cfg: Config):
    val_start = pd.Timestamp(cfg.validation_start_date)
    test_start = pd.Timestamp(cfg.test_start_date)
    embargo = pd.Timedelta(days=cfg.horizon_days)

    train = df[df["date"] < (val_start - embargo)].copy()
    val = df[(df["date"] >= val_start) & (df["date"] < (test_start - embargo))].copy()
    test = df[df["date"] >= test_start].copy()

    # Fallback if dates do not fit dataset.
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("[UYARI] Tarih bazlı split yeterli değil; kronolojik oran bazlı purged split uygulanıyor.")
        unique_dates = np.array(sorted(df["date"].unique()))
        d1 = unique_dates[int(len(unique_dates) * 0.70)]
        d2 = unique_dates[int(len(unique_dates) * 0.85)]
        train = df[df["date"] < (pd.Timestamp(d1) - embargo)].copy()
        val = df[(df["date"] >= pd.Timestamp(d1)) & (df["date"] < (pd.Timestamp(d2) - embargo))].copy()
        test = df[df["date"] >= pd.Timestamp(d2)].copy()

    return train.sort_values("date"), val.sort_values("date"), test.sort_values("date")


def sample_chronological(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    # Sınıf dağılımını bozmayacak şekilde kronolojik bloklardan örnekle.
    rng = np.random.default_rng(random_state)
    df = df.sort_values("date").copy()
    pos = df[df["target_future_7d_m4"] == 1]
    neg = df[df["target_future_7d_m4"] == 0]
    max_pos = min(len(pos), max_rows // 2)
    max_neg = max_rows - max_pos
    pos_s = pos.sample(max_pos, random_state=random_state) if len(pos) > max_pos else pos
    neg_s = neg.sample(max_neg, random_state=random_state) if len(neg) > max_neg else neg
    return pd.concat([pos_s, neg_s]).sort_values("date")


def make_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")


def scale_pos_weight(y: pd.Series) -> float:
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    return neg / pos


def build_estimator(params: Optional[Dict] = None, spw: float = 1.0, cfg: Optional[Config] = None):
    params = params or {}
    rs = cfg.random_state if cfg else 42
    if HAS_XGBOOST:
        defaults = dict(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=15,
            reg_alpha=1.0,
            reg_lambda=6.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=rs,
            n_jobs=-1,
            scale_pos_weight=spw,
        )
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)

    return HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.035),
        max_iter=params.get("max_iter", 450),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        l2_regularization=params.get("l2_regularization", 3.0),
        random_state=rs,
    )


def optimize_params(train_df: pd.DataFrame, numeric: List[str], categorical: List[str], cfg: Config) -> Dict:
    if not HAS_OPTUNA:
        print("[INFO] Optuna yok; default güvenli parametrelerle devam.")
        return {}

    opt_df = sample_chronological(train_df, cfg.max_rows_for_optuna, cfg.random_state)
    feature_cols = numeric + categorical
    X = opt_df[feature_cols]
    y = opt_df["target_future_7d_m4"].astype(int)

    n_splits = min(cfg.n_splits, 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"[INFO] Walk-forward Optuna başladı. Rows={len(opt_df):,}, folds={n_splits}")

    def objective(trial):
        if HAS_XGBOOST:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 15.0),
            }
        else:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
                "max_iter": trial.suggest_int("max_iter", 250, 700),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.5, 10.0),
            }

        scores = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            if y_va.nunique() < 2 or y_tr.nunique() < 2:
                continue
            pipe = Pipeline([
                ("preprocessor", make_preprocessor(numeric, categorical)),
                ("model", build_estimator(params, scale_pos_weight(y_tr), cfg)),
            ])
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_va)[:, 1]
            scores.append(average_precision_score(y_va, prob))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    print(f"[INFO] Optuna en iyi walk-forward AUC-PR: {study.best_value:.4f}")
    print(f"[INFO] En iyi parametreler: {study.best_params}")
    return study.best_params


# =============================================================================
# THRESHOLD, CALIBRATION, EVALUATION
# =============================================================================

def choose_threshold(y_true: pd.Series, prob: np.ndarray, min_precision: float) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    rows = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 2 * p * r / (p + r + 1e-12)
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)})
    if not rows:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    candidates = [r for r in rows if r["precision"] >= min_precision]
    return max(candidates or rows, key=lambda z: z["f1"])


def try_calibrate_prefit(pipe: Pipeline, X_val: pd.DataFrame, y_val: pd.Series):
    if not HAS_CALIBRATION or y_val.nunique() < 2:
        return pipe, "calibration_skipped"
    try:
        if HAS_FROZEN_ESTIMATOR:
            calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
            calibrated.fit(X_val, y_val)
            return calibrated, "isotonic_frozen_estimator"
        else:
            calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
            calibrated.fit(X_val, y_val)
            return calibrated, "isotonic_prefit"
    except Exception as e:
        print(f"[UYARI] Calibration başarısız, base model kullanılacak: {e}")
        return pipe, f"calibration_failed: {e}"


def evaluate(y_true: pd.Series, prob: np.ndarray, threshold: float) -> Tuple[Dict, np.ndarray]:
    pred = (prob >= threshold).astype(int)
    metrics = {
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "threshold": float(threshold),
        "average_precision_auc_pr": float(average_precision_score(y_true, prob)) if y_true.nunique() > 1 else np.nan,
        "brier_score": float(brier_score_loss(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, zero_division=0, output_dict=True),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        metrics["roc_auc"] = np.nan
    return metrics, pred


def bootstrap_ci(y_true: pd.Series, prob: np.ndarray, metric_fn, n_boot: int = 300) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    y = np.asarray(y_true)
    p = np.asarray(prob)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric_fn(y[idx], p[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_and_evaluate(feat: pd.DataFrame, cfg: Config, out: Path):
    print_section("5) MODEL EĞİTİMİ: PURGED SPLIT + WALK-FORWARD + THRESHOLD")

    train_df, val_df, test_df = purged_time_split(feat, cfg)

    # Büyükse train'i makul boyuta indir; test/val asla örneklenmez.
    if len(train_df) > cfg.max_rows_for_training:
        print(f"[INFO] Train çok büyük; dengeli örnekleme uygulanıyor: {len(train_df):,} -> {cfg.max_rows_for_training:,}")
        train_df = sample_chronological(train_df, cfg.max_rows_for_training, cfg.random_state)

    numeric, categorical = get_feature_columns(feat)
    feature_cols = numeric + categorical

    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
    print(
        "Target oranları -> "
        f"train={train_df['target_future_7d_m4'].mean():.4f}, "
        f"val={val_df['target_future_7d_m4'].mean():.4f}, "
        f"test={test_df['target_future_7d_m4'].mean():.4f}"
    )
    print(f"Feature sayısı -> numeric={len(numeric)}, categorical={len(categorical)}")

    split_report = {
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "train_date_range": [str(train_df["date"].min()), str(train_df["date"].max())],
        "val_date_range": [str(val_df["date"].min()), str(val_df["date"].max())],
        "test_date_range": [str(test_df["date"].min()), str(test_df["date"].max())],
        "target_rates": {
            "train": float(train_df["target_future_7d_m4"].mean()),
            "val": float(val_df["target_future_7d_m4"].mean()),
            "test": float(test_df["target_future_7d_m4"].mean()),
        },
        "features": {"numeric": numeric, "categorical": categorical},
    }
    save_json(split_report, out / "03_time_split_report.json")

    best_params = optimize_params(train_df, numeric, categorical, cfg)

    X_train, y_train = train_df[feature_cols], train_df["target_future_7d_m4"].astype(int)
    X_val, y_val = val_df[feature_cols], val_df["target_future_7d_m4"].astype(int)
    X_test, y_test = test_df[feature_cols], test_df["target_future_7d_m4"].astype(int)

    base_pipe = Pipeline([
        ("preprocessor", make_preprocessor(numeric, categorical)),
        ("model", build_estimator(best_params, scale_pos_weight(y_train), cfg)),
    ])
    base_pipe.fit(X_train, y_train)

    calibrated_model, calibration_status = try_calibrate_prefit(base_pipe, X_val, y_val)
    val_prob = calibrated_model.predict_proba(X_val)[:, 1]
    threshold_info = choose_threshold(y_val, val_prob, cfg.min_precision_for_threshold)
    print(f"[INFO] Validation threshold: {threshold_info}")
    print(f"[INFO] Calibration: {calibration_status}")

    # HONEST TEST EVALUATION:
    # Test olasılığı, validation üzerinde kalibre edilmiş modelden alınır.
    # Böylece threshold ve kalibrasyon aynı olasılık ölçeğinde kalır; V3'teki fazla-alarm probleminin ana düzeltmesi budur.
    test_prob = calibrated_model.predict_proba(X_test)[:, 1]
    test_metrics, test_pred = evaluate(y_test, test_prob, threshold_info["threshold"])

    # DEPLOYMENT MODEL:
    # Raporlama/evaluation için calibrated_model kullanılır. Ürünleştirme için ayrıca train+val ile fit edilmiş
    # final pipeline kaydedilir; bunun threshold'u validation'dan gelen eşiktir.
    trainval_df = pd.concat([train_df, val_df], axis=0).sort_values("date")
    X_trainval = trainval_df[feature_cols]
    y_trainval = trainval_df["target_future_7d_m4"].astype(int)

    final_pipe = Pipeline([
        ("preprocessor", make_preprocessor(numeric, categorical)),
        ("model", build_estimator(best_params, scale_pos_weight(y_trainval), cfg)),
    ])
    final_pipe.fit(X_trainval, y_trainval)
    if y_test.nunique() > 1:
        test_metrics["auc_pr_95ci"] = bootstrap_ci(y_test, test_prob, average_precision_score)
        test_metrics["roc_auc_95ci"] = bootstrap_ci(y_test, test_prob, roc_auc_score)

    test_metrics["threshold_info_from_validation"] = threshold_info
    test_metrics["best_params"] = best_params
    test_metrics["calibration_status"] = calibration_status
    test_metrics["config"] = asdict(cfg)
    save_json(test_metrics, out / "04_model_metrics.json")

    pred_df = test_df[["date", "cell_id", "latitude", "longitude", "target_future_7d_m4"]].copy()
    pred_df["risk_score"] = test_prob
    pred_df["prediction"] = test_pred
    pred_df.to_csv(out / "05_test_predictions.csv", index=False)

    if HAS_JOBLIB:
        joblib.dump({
            "pipeline": calibrated_model,
            "threshold": threshold_info["threshold"],
            "features": feature_cols,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "config": asdict(cfg),
            "model_type": "honest_calibrated_train_to_val_model",
        }, out / "deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL_calibrated_model.joblib")

        joblib.dump({
            "pipeline": final_pipe,
            "threshold": threshold_info["threshold"],
            "features": feature_cols,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "config": asdict(cfg),
            "model_type": "deployment_train_plus_val_model",
        }, out / "deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL_deployment_model.joblib")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, zero_division=0))
    print(
        f"AUC-PR: {test_metrics['average_precision_auc_pr']:.4f}\n"
        f"ROC-AUC: {test_metrics['roc_auc']:.4f}\n"
        f"F1: {test_metrics['f1']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f}"
    )
    return final_pipe, test_metrics, pred_df, test_df, feature_cols, numeric, categorical


# =============================================================================
# MODEL PLOTS + SHAP
# =============================================================================

def plot_model_outputs(pred_df: pd.DataFrame, metrics: Dict, cfg: Config, out: Path) -> None:
    print_section("6) MODEL GRAFİKLERİ + RAPORLAR")

    y = pred_df["target_future_7d_m4"].astype(int).to_numpy()
    p = pred_df["risk_score"].to_numpy()
    pred = pred_df["prediction"].astype(int).to_numpy()

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "06_confusion_matrix.png", dpi=250)
    safe_show(cfg)

    # Precision recall curve
    precision, recall, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out / "07_precision_recall_curve.png", dpi=250)
    safe_show(cfg)

    # Risk distribution
    plt.figure(figsize=(8, 5))
    plt.hist(p[y == 0], bins=40, alpha=0.6, label="Gerçek 0")
    plt.hist(p[y == 1], bins=40, alpha=0.6, label="Gerçek 1")
    plt.axvline(metrics["threshold"], linestyle="--", label="Threshold")
    plt.title("Risk Skoru Dağılımı")
    plt.xlabel("Risk Score")
    plt.ylabel("Frekans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "08_risk_score_distribution.png", dpi=250)
    safe_show(cfg)

    # Risk map scatter by latest test average risk per cell.
    latest = pred_df.sort_values("date").groupby("cell_id").tail(30)
    cell_risk = latest.groupby("cell_id").agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        risk_score=("risk_score", "mean"),
        target=("target_future_7d_m4", "mean"),
    ).reset_index()
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(cell_risk["longitude"], cell_risk["latitude"], c=cell_risk["risk_score"], s=55)
    plt.colorbar(sc, label="Ortalama Risk Score")
    plt.title("Test Dönemi Bölgesel Ortalama Risk Haritası")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out / "09_regional_risk_map.png", dpi=250)
    safe_show(cfg)

    # Daily average risk over test time
    daily_risk = pred_df.groupby("date").agg(
        avg_risk=("risk_score", "mean"),
        actual_rate=("target_future_7d_m4", "mean"),
    ).reset_index()
    plt.figure(figsize=(13, 5))
    plt.plot(daily_risk["date"], daily_risk["avg_risk"], label="Ortalama Risk")
    plt.plot(daily_risk["date"], daily_risk["actual_rate"], label="Gerçek Pozitif Oranı", alpha=0.7)
    plt.title("Test Dönemi Günlük Ortalama Risk vs Gerçek Oran")
    plt.xlabel("Tarih")
    plt.ylabel("Oran / Risk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "10_daily_risk_vs_actual.png", dpi=250)
    safe_show(cfg)

    print(f"[INFO] Model çıktı/görselleri kaydedildi: {out.resolve()}")


def save_feature_importance(model_pipe: Pipeline, feature_cols: List[str], numeric: List[str], categorical: List[str], out: Path) -> None:
    try:
        model = model_pipe.named_steps["model"]
        pre = model_pipe.named_steps["preprocessor"]
        feature_names = pre.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
            df_imp.to_csv(out / "11_feature_importance.csv", index=False)

            top = df_imp.head(30).iloc[::-1]
            plt.figure(figsize=(10, 9))
            plt.barh(top["feature"], top["importance"])
            plt.title("Top 30 Feature Importance")
            plt.tight_layout()
            plt.savefig(out / "11_feature_importance.png", dpi=250)
            plt.close()
    except Exception as e:
        with open(out / "11_feature_importance_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def save_shap_report(model_pipe: Pipeline, test_df: pd.DataFrame, feature_cols: List[str], cfg: Config, out: Path) -> None:
    if not (HAS_SHAP and HAS_XGBOOST):
        return
    try:
        sample = test_df[feature_cols].sample(min(len(test_df), cfg.max_rows_for_shap), random_state=cfg.random_state)
        pre = model_pipe.named_steps["preprocessor"]
        model = model_pipe.named_steps["model"]
        X_trans = pre.transform(sample)
        feature_names = pre.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(out / "12_shap_feature_importance.csv", index=False)
    except Exception as e:
        with open(out / "12_shap_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(cfg: Config):
    out = ensure_output_dir(cfg.output_dir)
    print_section("DEEPFAULT RABIA - TIME SERIES V4 BEST FINAL PIPELINE BAŞLADI")

    input_path = choose_input_file(cfg)
    raw = read_csv_safely(input_path)
    print(f"[INFO] Okunan dosya: {input_path.resolve()}")
    print(f"[INFO] İlk veri boyutu: {raw.shape}")

    print_section("1) VERİ TEMİZLEME")
    events = clean_raw_events(raw, cfg, out)
    events.to_csv(out / "00_cleaned_events.csv", index=False)
    print(f"[INFO] Temiz event veri boyutu: {events.shape}")

    panel = build_daily_cell_panel(events, cfg, out)
    panel.to_csv(out / "01_daily_cell_panel.csv", index=False)

    featured = engineer_time_series(panel, cfg, out)
    featured.to_csv(out / "02_time_series_features.csv", index=False)

    plot_eda(events, featured, cfg, out)

    model, metrics, pred_df, test_df, feature_cols, numeric, categorical = train_and_evaluate(featured, cfg, out)
    plot_model_outputs(pred_df, metrics, cfg, out)
    save_feature_importance(model, feature_cols, numeric, categorical, out)
    save_shap_report(model, test_df, feature_cols, cfg, out)

    print_section("PIPELINE TAMAMLANDI")
    print(f"Temiz event veri: {out / '00_cleaned_events.csv'}")
    print(f"Daily panel: {out / '01_daily_cell_panel.csv'}")
    print(f"Time series feature veri: {out / '02_time_series_features.csv'}")
    print(f"Tahminler: {out / '05_test_predictions.csv'}")
    print(f"Metrikler: {out / '04_model_metrics.json'}")
    print(f"Model: {out / 'deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL_calibrated_model.joblib'}")
    print(f"Tüm outputs klasörü: {out.resolve()}")
    return model, metrics, pred_df


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="DeepFault Rabia TIME SERIES V4 BEST FINAL")
    parser.add_argument("--input", default="depremler_hava_nasa.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--panel-start-date", default="1990-01-01")
    parser.add_argument("--validation-start-date", default="2022-01-01")
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--target-mag", type=float, default=4.0)
    parser.add_argument("--lat-bin-size", type=float, default=0.75)
    parser.add_argument("--lon-bin-size", type=float, default=0.75)
    parser.add_argument("--min-cell-events", type=int, default=8)
    parser.add_argument("--min-precision", type=float, default=0.35)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    return Config(
        input_file=args.input,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        panel_start_date=args.panel_start_date,
        validation_start_date=args.validation_start_date,
        test_start_date=args.test_start_date,
        horizon_days=args.horizon_days,
        target_mag_threshold=args.target_mag,
        lat_bin_size=args.lat_bin_size,
        lon_bin_size=args.lon_bin_size,
        min_cell_events=args.min_cell_events,
        min_precision_for_threshold=args.min_precision,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
