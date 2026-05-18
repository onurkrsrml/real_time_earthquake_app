"""Profesyonel UI — DeepFault marka teması."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from deepfault.config import APP_SUBTITLE, APP_TITLE
from deepfault.paths import LOGO_PATH

BLUE = "#00B4D8"
BLUE_DEEP = "#0077B6"
ORANGE = "#FF8500"
ORANGE_DEEP = "#E85D04"
TEXT = "#E8EDF4"
MUTED = "#8B9CB3"


def _logo_base64() -> str:
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            #MainMenu, footer, header {{visibility: hidden;}}
            .stApp {{
                background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a2840 0%, #0a0e17 45%, #06080f 100%);
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #0f1624 0%, #0a0e17 100%);
                border-right: 1px solid rgba(0,180,216,0.12);
            }}
            [data-testid="stSidebar"] .stMarkdown h2 {{
                color: {TEXT};
                font-weight: 700;
            }}
            h1, h2, h3, h4 {{
                color: {TEXT} !important;
                font-family: 'Inter', sans-serif !important;
                letter-spacing: -0.02em;
            }}
            p, label, span, .stMarkdown {{
                color: {MUTED};
                font-family: 'Inter', sans-serif;
            }}
            div[data-testid="stMetric"] {{
                background: linear-gradient(145deg, rgba(0,119,182,0.12) 0%, rgba(255,133,0,0.06) 100%);
                border: 1px solid rgba(0,180,216,0.2);
                border-radius: 14px;
                padding: 14px 18px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.25);
            }}
            div[data-testid="stMetric"] label {{
                color: {MUTED} !important;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: {BLUE} !important;
                font-weight: 700;
                font-size: 1.6rem;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 6px;
                background: rgba(255,255,255,0.03);
                border-radius: 12px;
                padding: 6px;
                border: 1px solid rgba(255,255,255,0.06);
            }}
            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px;
                color: {MUTED};
                font-weight: 500;
                padding: 10px 18px;
            }}
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(90deg, rgba(0,180,216,0.25), rgba(255,133,0,0.15)) !important;
                color: {TEXT} !important;
                border: 1px solid rgba(0,180,216,0.35) !important;
            }}
            .stButton > button[kind="primary"] {{
                background: linear-gradient(90deg, {BLUE_DEEP}, {ORANGE});
                border: none;
                font-weight: 600;
                border-radius: 10px;
            }}
            .stProgress > div > div {{
                background: linear-gradient(90deg, {BLUE}, {ORANGE});
            }}
            [data-testid="stDataFrame"] {{
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
            }}
            .df-header {{
                display: flex;
                align-items: center;
                gap: 1.25rem;
                padding: 1rem 0 1.5rem 0;
                border-bottom: 1px solid rgba(0,180,216,0.15);
                margin-bottom: 1.25rem;
            }}
            .df-header img {{
                height: 72px;
                width: auto;
                filter: drop-shadow(0 0 20px rgba(0,180,216,0.35));
            }}
            .df-header-text h1 {{
                margin: 0;
                font-size: 1.75rem;
                font-weight: 800;
                background: linear-gradient(90deg, {BLUE}, {ORANGE});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: 0.04em;
            }}
            .df-header-text p {{
                margin: 0.25rem 0 0;
                font-size: 0.8rem;
                color: {MUTED};
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}
            .df-section {{
                background: rgba(15,22,35,0.65);
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                backdrop-filter: blur(8px);
            }}
            .df-section-title {{
                font-size: 1.1rem;
                font-weight: 600;
                color: {TEXT};
                margin: 0 0 0.35rem 0;
            }}
            .df-section-sub {{
                font-size: 0.85rem;
                color: {MUTED};
                margin: 0 0 1rem 0;
            }}
            .df-badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 0.72rem;
                font-weight: 600;
                margin-right: 6px;
            }}
            .df-badge-blue {{
                background: rgba(0,180,216,0.15);
                color: {BLUE};
                border: 1px solid rgba(0,180,216,0.3);
            }}
            .df-badge-orange {{
                background: rgba(255,133,0,0.12);
                color: {ORANGE};
                border: 1px solid rgba(255,133,0,0.3);
            }}
            .df-disclaimer {{
                background: rgba(255,133,0,0.08);
                border-left: 3px solid {ORANGE};
                padding: 12px 16px;
                border-radius: 0 10px 10px 0;
                font-size: 0.88rem;
                color: {MUTED};
            }}
            .df-model-title {{
                font-size: 0.95rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin: 0 0 0.75rem 0;
            }}
            .df-commentary-box {{
                background: linear-gradient(135deg, rgba(0,119,182,0.1) 0%, rgba(255,133,0,0.06) 100%);
                border: 1px solid rgba(0,180,216,0.25);
                border-radius: 14px;
                padding: 1.25rem 1.5rem;
                margin-top: 0.5rem;
            }}
            .df-map-legend {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 0.8rem;
                color: {MUTED};
                margin-bottom: 0.75rem;
            }}
            .df-map-legend span {{
                display: inline-block;
                width: 14px;
                height: 14px;
                border-radius: 50%;
                vertical-align: middle;
            }}
            [data-testid="stFolium"] {{
                border: 1px solid rgba(0,180,216,0.2);
                border-radius: 14px;
                overflow: hidden;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    b64 = _logo_base64()
    img_tag = (
        f'<img src="data:image/png;base64,{b64}" alt="DeepFault Logo"/>' if b64 else ""
    )
    st.markdown(
        f"""
        <div class="df-header">
            {img_tag}
            <div class="df-header-text">
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE} · TÜRKİYE</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str = "") -> None:
    sub = f'<p class="df-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="df-section-title">{title}</div>
        {sub}
        """,
        unsafe_allow_html=True,
    )


def render_badges() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
            <span class="df-badge df-badge-blue">XGBoost · Rabia</span>
            <span class="df-badge df-badge-orange">Regresyon · Onur</span>
            <span class="df-badge df-badge-blue">USGS + NASA POWER</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
