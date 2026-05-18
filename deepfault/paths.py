"""Proje kök yolları."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "depremler_hava_nasa.csv"
PREDICTIONS_CSV = ROOT / "outputs" / "05_test_predictions.csv"
MODEL_METRICS_JSON = ROOT / "outputs" / "04_model_metrics.json"
RABIA_MODEL = ROOT / "outputs" / "deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL_calibrated_model.joblib"
ONUR_MODEL_MAG = ROOT / "onur" / "model_deepfault_mag.pkl"
ONUR_MODEL_DAYS = ROOT / "onur" / "model_deepfault_days.pkl"
ONUR_METADATA = ROOT / "onur" / "OUTPUTS_onur.json"
LOGO_PATH = PACKAGE_DIR / "assets" / "deepfault_logo.png"

LAT_BIN = 0.75
LON_BIN = 0.75
STRONG_MAG = 4.0
DEFAULT_RADIUS_KM = 75
