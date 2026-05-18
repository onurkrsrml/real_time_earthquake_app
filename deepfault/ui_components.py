"""Yeniden kullanılabilir UI bileşenleri."""

from __future__ import annotations

import html
import json

import streamlit as st

from deepfault.styles import BLUE, ORANGE


def render_kpi_row(inference: dict) -> None:
    """Üst KPI şeridi — tek satır."""
    rabia = inference.get("rabia", {})
    onur = inference.get("onur", {})
    agg = inference.get("province_aggregate") or {}
    combined = inference.get("combined_risk_score")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Birleşik Risk", f"{combined:.1f}" if combined is not None else "—")
    if rabia.get("ok"):
        c2.metric("XGBoost Olasılık", f"%{rabia['risk_probability'] * 100:.1f}")
    else:
        c2.metric("XGBoost", "—")
    if onur.get("ok"):
        c3.metric("Max M (7g)", f"{onur['pred_max_magnitude_7d']:.2f}")
    else:
        c3.metric("Onur Regresyon", "—")
    if not agg.get("empty"):
        c4.metric("Grid Ort.", f"%{agg['mean_probability'] * 100:.1f}")
    else:
        c4.metric("Grid Ort.", "—")


def render_rabia_card(rabia: dict) -> None:
    st.markdown(
        f'<p class="df-model-title" style="color:{BLUE};">Rabia · XGBoost (Kalibre)</p>',
        unsafe_allow_html=True,
    )
    if not rabia.get("ok"):
        st.error(rabia.get("error", "Çıktı üretilemedi."))
        return

    st.markdown(
        f"**Risk seviyesi:** {rabia.get('icon', '')} {rabia.get('risk_level', '—')}  \n"
        f"**7 gün M≥4 olasılığı:** %{rabia['risk_probability'] * 100:.1f}  \n"
        f"**Skor:** {rabia['risk_score']}/100  \n"
        f"**Grid hücresi:** `{rabia.get('cell_id', '—')}`"
    )

    with st.expander("Ham model çıktısı (JSON)", expanded=False):
        st.code(
            json.dumps(
                {
                    k: rabia[k]
                    for k in (
                        "risk_probability",
                        "risk_score",
                        "prediction",
                        "threshold",
                        "cell_id",
                        "feature_end_date",
                    )
                    if k in rabia
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


def render_onur_card(onur: dict) -> None:
    st.markdown(
        f'<p class="df-model-title" style="color:{ORANGE};">Onur · Regresyon</p>',
        unsafe_allow_html=True,
    )
    if not onur.get("ok"):
        st.error(onur.get("error", "Çıktı üretilemedi."))
        return

    st.markdown(
        f"**Tahmini max büyüklük (7g):** {onur['pred_max_magnitude_7d']:.2f}  \n"
        f"**Büyük olaya kalan gün:** {onur['pred_days_to_major']:.1f}  \n"
        f"**Güven:** {onur['confidence']:.2f}  \n"
        f"**Yakın olay sayısı:** {onur.get('n_nearby_events', '—')}"
    )

    with st.expander("Ham model çıktısı (JSON)", expanded=False):
        st.code(
            json.dumps(
                {
                    k: onur[k]
                    for k in (
                        "pred_max_magnitude_7d",
                        "pred_days_to_major",
                        "confidence",
                        "n_nearby_events",
                        "date_range",
                        "reference_event_time",
                    )
                    if k in onur
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


def render_commentary_box(commentary: str) -> None:
    body = html.escape(commentary).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="df-commentary-box">
            <h4 style="margin:0 0 0.75rem 0;color:#E8EDF4;">Ajan yorum özeti</h4>
            <p style="margin:0;font-size:0.95rem;line-height:1.55;color:#B8C5D6;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_map_legend() -> None:
    st.markdown(
        """
        <div class="df-map-legend">
            <span style="background:#2e7d32;"></span> Düşük &nbsp;
            <span style="background:#f9a825;"></span> Orta &nbsp;
            <span style="background:#e65100;"></span> Yüksek &nbsp;
            <span style="background:#b71c1c;"></span> Çok yüksek
        </div>
        """,
        unsafe_allow_html=True,
    )
