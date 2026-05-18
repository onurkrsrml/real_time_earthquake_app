"""Streamlit oturum durumu."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from deepfault.config import TURKEY_PROVINCES


def _default_dates() -> tuple[date, date]:
    try:
        from deepfault.data_bounds import prediction_date_bounds

        dmin, dmax = prediction_date_bounds()
        end = dmax
        start = max(dmin, end - timedelta(days=30))
        return start, end
    except Exception:
        end = date.today()
        return end - timedelta(days=30), end


def init_session_state() -> None:
    start_def, end_def = _default_dates()
    defaults = {
        "selected_province": "İstanbul",
        "location_mode": "İl seç",
        "custom_lat": 41.0082,
        "custom_lon": 28.9784,
        "date_range_start": start_def,
        "date_range_end": end_def,
        "sidebar_date_range": (start_def, end_def),
        "last_inference": None,
        "last_risk_score": None,
        "model_output_summary": "",
        "agent_commentary": "",
        "inference_location_key": None,
        "webhook_url": "",
        "telegram_chat_id": "",
        "telegram_bot_token": "",
        "user_note": "",
        "radius_km": 75,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_province_coords(province: str) -> None:
    if province in TURKEY_PROVINCES:
        lat, lon = TURKEY_PROVINCES[province]
        st.session_state.custom_lat = lat
        st.session_state.custom_lon = lon
        st.session_state.selected_province = province


def get_date_range() -> tuple[date, date]:
    from deepfault.data_bounds import clamp_date_range

    start = st.session_state.get("date_range_start")
    end = st.session_state.get("date_range_end")
    if start is None or end is None:
        start, end = _default_dates()
    if start > end:
        start, end = end, start
    start, end = clamp_date_range(start, end)
    st.session_state.date_range_start = start
    st.session_state.date_range_end = end
    return start, end


def current_inference_key() -> str:
    """Konum + yarıçap + tarih aralığı değişince çıkarım yenilenir."""
    start, end = get_date_range()
    radius = st.session_state.get("radius_km", 75)
    if st.session_state.get("location_mode") == "Koordinat gir":
        loc = f"coord:{st.session_state.custom_lat:.4f}:{st.session_state.custom_lon:.4f}"
    else:
        loc = f"prov:{st.session_state.get('selected_province', '')}"
    return f"{loc}|r{radius}|{start.isoformat()}|{end.isoformat()}"


def get_active_coords() -> tuple[str | None, float, float]:
    if st.session_state.get("location_mode") == "Koordinat gir":
        return (
            None,
            float(st.session_state.custom_lat),
            float(st.session_state.custom_lon),
        )
    prov = st.session_state.get("selected_province", "İstanbul")
    sync_province_coords(prov)
    return prov, float(st.session_state.custom_lat), float(st.session_state.custom_lon)
