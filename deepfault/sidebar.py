"""Sol sidebar."""

from __future__ import annotations

import streamlit as st

from deepfault.agent_commentary import build_agent_commentary
from deepfault.config import TURKEY_PROVINCES
from deepfault.footer import render_footer
from deepfault.inference import run_live_inference
from deepfault.notifications import build_agent_payload, send_webhook_telegram
from deepfault.session_state import (
    current_inference_key,
    get_active_coords,
    get_date_range,
    sync_province_coords,
)
from deepfault.styles import BLUE, MUTED, ORANGE


def ui_sidebar() -> None:
    st.sidebar.markdown(
        f'<p style="color:{MUTED};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin:0;">Kontrol Paneli</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(f"## <span style='color:{BLUE}'>Deep</span><span style='color:{ORANGE}'>Fault</span>", unsafe_allow_html=True)

    provinces = sorted(TURKEY_PROVINCES.keys())
    current = st.session_state.get("selected_province", "İstanbul")
    prov_index = provinces.index(current) if current in provinces else 0

    province = st.sidebar.selectbox("İl", provinces, index=prov_index, key="sidebar_province_select")
    sync_province_coords(province)

    st.sidebar.radio(
        "Konum modu",
        ["İl seç", "Koordinat gir"],
        index=0 if st.session_state.get("location_mode") == "İl seç" else 1,
        key="sidebar_location_mode",
    )

    if st.session_state.location_mode == "Koordinat gir":
        st.sidebar.number_input("Enlem", format="%.4f", key="custom_lat")
        st.sidebar.number_input("Boylam", format="%.4f", key="custom_lon")
    else:
        sync_province_coords(province)

    st.sidebar.slider("Agregasyon yarıçapı (km)", 25, 150, key="radius_km")

    from deepfault.data_bounds import prediction_date_bounds

    dmin, dmax = prediction_date_bounds()
    dr = st.sidebar.date_input(
        "Tarih aralığı",
        key="sidebar_date_range",
        min_value=dmin,
        max_value=dmax,
    )
    if isinstance(dr, tuple) and len(dr) == 2:
        st.session_state.date_range_start, st.session_state.date_range_end = dr

    st.sidebar.caption(f"Veri kapsamı: {dmin} → {dmax}")

    _refresh_inference_if_needed()

    st.sidebar.divider()
    _render_notification_section()
    st.sidebar.divider()
    render_footer(sidebar=True)


def _date_range_str() -> str:
    start, end = get_date_range()
    return f"{start.isoformat()} — {end.isoformat()}"


def _refresh_inference_if_needed() -> None:
    key = current_inference_key()
    if key == st.session_state.get("inference_location_key") and st.session_state.get("last_inference"):
        _update_commentary()
        return

    province, lat, lon = get_active_coords()
    start, end = get_date_range()

    with st.spinner("Modeller analiz ediliyor…"):
        try:
            result = run_live_inference(
                province,
                lat,
                lon,
                radius_km=float(st.session_state.radius_km),
                date_start=start,
                date_end=end,
            )
            st.session_state.last_inference = result
            st.session_state.last_risk_score = result.get("combined_risk_score")
            st.session_state.inference_location_key = key
        except Exception as exc:
            st.sidebar.error(f"Çıkarım hatası: {exc}")

    _update_commentary()


def _update_commentary() -> None:
    inference = st.session_state.get("last_inference")
    if not inference:
        return
    province, lat, lon = get_active_coords()
    commentary = build_agent_commentary(
        inference,
        province,
        lat,
        lon,
        st.session_state.get("user_note", ""),
        _date_range_str(),
    )
    st.session_state.agent_commentary = commentary
    st.session_state.model_output_summary = commentary


def _render_notification_section() -> None:
    st.sidebar.subheader("Bildirim & Raporlama")
    for k in ("webhook_url", "telegram_chat_id", "telegram_bot_token", "user_note"):
        if k not in st.session_state:
            st.session_state[k] = ""

    st.sidebar.text_input("Webhook URL", key="webhook_url", placeholder="https://your-agent/webhook")
    st.sidebar.text_input("Telegram Chat ID", key="telegram_chat_id")
    st.sidebar.text_input("Telegram Bot Token", key="telegram_bot_token", type="password")
    st.sidebar.text_area("Kullanıcı Notu", key="user_note", height=80)

    if st.sidebar.button("Ajan'a Gönder", type="primary", use_container_width=True):
        _handle_send_to_agent()


def _handle_send_to_agent() -> None:
    _refresh_inference_if_needed()
    inference = st.session_state.get("last_inference") or {}
    province, lat, lon = get_active_coords()
    location = f"{province or 'Koordinat'} ({lat:.4f}, {lon:.4f})"
    commentary = build_agent_commentary(
        inference, province, lat, lon, st.session_state.user_note, _date_range_str()
    )
    payload = build_agent_payload(
        risk_score=st.session_state.get("last_risk_score"),
        selected_city_or_coordinate=location,
        date_range=_date_range_str(),
        model_output_summary=commentary,
        user_note=st.session_state.user_note,
    )
    with st.spinner("Rapor gönderiliyor…"):
        results = send_webhook_telegram(
            st.session_state.webhook_url,
            st.session_state.telegram_bot_token,
            st.session_state.telegram_chat_id,
            payload,
        )
    for channel, ok, msg in results:
        if ok:
            st.sidebar.success(f"{channel}: {msg}")
        else:
            st.sidebar.error(f"{channel}: {msg}")
