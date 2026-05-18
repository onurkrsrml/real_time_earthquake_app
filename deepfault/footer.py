"""Geliştirici footer bileşeni."""

from __future__ import annotations

import streamlit as st

from deepfault.config import DEVELOPERS


def render_footer(sidebar: bool = False) -> None:
    names = " · ".join(DEVELOPERS)
    html = f"""
    <div class="df-footer">
        <div style="letter-spacing:0.08em;text-transform:uppercase;font-size:0.7rem;margin-bottom:0.35rem;">
            DeepFault Geliştirici Ekibi
        </div>
        <strong>{names}</strong>
    </div>
    """
    if sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)
