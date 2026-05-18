"""Veri dosyalarının tarih sınırları."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from deepfault.paths import PREDICTIONS_CSV


@st.cache_data(show_spinner=False)
def prediction_date_bounds() -> tuple[date, date]:
    df = pd.read_csv(PREDICTIONS_CSV, usecols=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df["date"].min().date(), df["date"].max().date()


def clamp_date_range(start: date, end: date) -> tuple[date, date]:
    """Kullanıcı tarihini tahmin verisi aralığına sıkıştırır."""
    dmin, dmax = prediction_date_bounds()
    end = min(end, dmax)
    start = max(start, dmin)
    if start > end:
        start = max(dmin, end - timedelta(days=30))
    return start, end
