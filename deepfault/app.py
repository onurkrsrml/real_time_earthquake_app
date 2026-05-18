"""
DeepFault — AI-Powered Earthquake Risk & Early Warning System

Çalıştırma:
    streamlit run deepfault/app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepfault.agent_commentary import build_agent_commentary  # noqa: E402
from deepfault.analytics import (  # noqa: E402
    filter_events_by_date,
    map_cells_in_range,
    regional_risk_timeseries,
    seismic_stats_near,
)
from deepfault.charts import (  # noqa: E402
    chart_has_data,
    fig_daily_events,
    fig_magnitude_distribution,
    fig_metrics_summary,
    fig_model_metrics,
    fig_risk_timeseries,
    fig_top_risk_cells,
)
from deepfault.config import APP_ICON, APP_SUBTITLE, APP_TITLE  # noqa: E402
from deepfault.footer import render_footer  # noqa: E402
from deepfault.geo import filter_near, haversine_km  # noqa: E402
from deepfault.inference import load_model_metrics, load_predictions, load_raw_events  # noqa: E402
from deepfault.maps import (  # noqa: E402
    build_risk_folium_map,
    build_risk_plotly_scatter_geo,
    prepare_map_data,
)
from deepfault.paths import LOGO_PATH  # noqa: E402
from deepfault.session_state import get_active_coords, get_date_range, init_session_state  # noqa: E402
from deepfault.sidebar import ui_sidebar  # noqa: E402
from deepfault.styles import inject_global_styles, render_badges, render_header, render_section  # noqa: E402
from deepfault.ui_components import (  # noqa: E402
    render_commentary_box,
    render_kpi_row,
    render_map_legend,
    render_onur_card,
    render_rabia_card,
)


@st.cache_data(show_spinner=False)
def _cached_map_df(date_start_iso: str, date_end_iso: str) -> pd.DataFrame:
    pred_df = load_predictions()
    return prepare_map_data(
        pred_df,
        date_start=date.fromisoformat(date_start_iso),
        date_end=date.fromisoformat(date_end_iso),
    )


def _safe_plotly(fig) -> None:
    if fig is None:
        return
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.error(f"Grafik hatası: {exc}")


def _render_chart_grid(charts: list) -> bool:
    """Yalnızca verisi olan grafikleri 2 sütunda gösterir. En az bir grafik çizildiyse True."""
    if not charts:
        return False
    for i in range(0, len(charts), 2):
        if i + 1 < len(charts):
            left, right = st.columns(2)
            with left:
                _safe_plotly(charts[i])
            with right:
                _safe_plotly(charts[i + 1])
        else:
            _safe_plotly(charts[i])
    return True


def _tab_home() -> None:
    render_section("Sistem Özeti", "Olasılıksal kısa vadeli sismik risk — kesin deprem tahmini değildir.")
    render_badges()

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(
            """
            **DeepFault**, Türkiye ve çevresindeki aktif fay zonlarında uzaysal-zamansal
            özelliklerle **7 günlük** sismik risk yoğunluğunu istatistiksel olarak modelleyen
            bir yapay zeka platformudur.

            - **Veri:** USGS · NASA POWER · astronomik değişkenler
            - **Sınıflandırma:** XGBoost (kalibre) — M≥4 olasılığı
            - **Regresyon:** Büyüklük ve büyük olaya kalan gün tahmini
            """
        )
        st.markdown(
            '<div class="df-disclaimer">'
            "Resmi deprem erken uyarı sistemi değildir. Acil durumlarda AFAD ve yerel yetkilileri izleyin."
            "</div>",
            unsafe_allow_html=True,
        )
    with col_b:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width="stretch")

    inf = st.session_state.get("last_inference")
    if inf:
        render_kpi_row(inf)


def _tab_risk_map(pred_df: pd.DataFrame) -> None:
    render_section(
        "Risk Haritası",
        "Folium HeatMap + Plotly scatter_geo — sidebar tarih aralığı ve yarıçapına duyarlı",
    )

    province, lat, lon = get_active_coords()
    start, end = get_date_range()
    radius = float(st.session_state.radius_km)

    map_df = _cached_map_df(start.isoformat(), end.isoformat())
    if map_df.empty:
        st.warning(
            f"**{start}** — **{end}** aralığında harita verisi yok. "
            "Sidebar’dan tarih aralığını genişletin (veri: 2024-01-01 → 2026-03-18)."
        )
        return

    render_map_legend()

    agg = st.session_state.get("last_inference", {}).get("province_aggregate", {})
    m1, m2, m3, m4 = st.columns(4)
    if not agg.get("empty"):
        m1.metric(f"{province or 'Bölge'} — ort.", f"%{agg['mean_probability'] * 100:.1f}")
        m2.metric("Maks", f"%{agg['max_probability'] * 100:.1f}")
        m3.metric("Yarıçap", f"{int(radius)} km")
        m4.metric("Hücre", agg.get("n_cells", "—"))

    st.markdown("##### Folium — HeatMap")
    folium_map = build_risk_folium_map(
        map_df,
        center_lat=lat,
        center_lon=lon,
        highlight_lat=lat,
        highlight_lon=lon,
        province_label=province or "Seçili bölge",
        radius_km=radius,
    )
    st_folium(
        folium_map,
        width="stretch",
        height=520,
        returned_objects=[],
        key=f"risk_map_{start}_{end}_{lat:.2f}_{lon:.2f}_{radius}",
    )

    st.markdown("##### Plotly — scatter_geo")
    scatter_fig = build_risk_plotly_scatter_geo(map_df)
    if scatter_fig is not None:
        _safe_plotly(scatter_fig)

    with st.expander("En yüksek riskli hücreler (tablo)", expanded=False):
        cols = [c for c in ["cell_id", "risk_score", "max_risk", "n_days"] if c in map_df.columns]
        if cols:
            display = map_df.sort_values("risk_score", ascending=False)[cols].head(12).copy()
            display["risk_score"] = (display["risk_score"] * 100).round(1)
            display = display.rename(columns={"risk_score": "risk_%"})
            st.dataframe(display, width="stretch", hide_index=True)


def _tab_model_outputs(pred_df: pd.DataFrame, events: pd.DataFrame) -> None:
    render_section("Model Çıktıları", "Bölgesel zaman serisi, sismik istatistikler ve test metrikleri")

    province, lat, lon = get_active_coords()
    start, end = get_date_range()
    radius = float(st.session_state.radius_km)

    ts = regional_risk_timeseries(pred_df, lat, lon, radius, start, end)
    local_events = filter_events_by_date(filter_near(events, lat, lon, radius), start, end)
    cells = map_cells_in_range(pred_df, start, end)
    if not cells.empty:
        dist = haversine_km(lat, lon, cells["latitude"].to_numpy(), cells["longitude"].to_numpy())
        cells = cells.loc[dist <= radius].copy()
    stats = seismic_stats_near(events, lat, lon, radius, start, end)

    if not stats.get("empty"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deprem (yakın)", stats["event_count"])
        c2.metric("Ort. M", f"{stats['mean_magnitude']:.2f}")
        c3.metric("Maks M", f"{stats['max_magnitude']:.2f}")
        c4.metric("M≥4", stats["strong_events"])

    regional_charts: list = []
    if chart_has_data("risk_timeseries", ts):
        regional_charts.append(fig_risk_timeseries(ts))
    if chart_has_data("magnitude", local_events):
        regional_charts.append(fig_magnitude_distribution(local_events))
    if chart_has_data("top_cells", cells):
        regional_charts.append(fig_top_risk_cells(cells))
    if chart_has_data("daily_events", local_events):
        regional_charts.append(fig_daily_events(local_events))

    if regional_charts:
        st.markdown(
            f"**{province or 'Seçili bölge'}** · {start} → {end} · {int(radius)} km"
        )
        _render_chart_grid(regional_charts)
    else:
        st.info(
            f"**{province or 'Seçili bölge'}** için **{start} — {end}** aralığında "
            f"({int(radius)} km) görüntülenecek grafik verisi bulunamadı. "
            "Tarih aralığını veya yarıçapı değiştirmeyi deneyin."
        )

    try:
        metrics = load_model_metrics()
        if chart_has_data("metrics", metrics):
            st.markdown("##### Genel model performansı (test seti)")
            _render_chart_grid([fig_metrics_summary(metrics), fig_model_metrics(metrics)])
    except Exception as exc:
        st.caption(f"Model metrikleri: {exc}")


def _tab_live_inference() -> None:
    render_section("Canlı Çıkarım", "Rabia XGBoost + Onur regresyon — anlık risk değerlendirmesi")

    province, lat, lon = get_active_coords()
    start, end = get_date_range()

    loc_line = f"**{province or 'Özel koordinat'}** · {lat:.4f}°, {lon:.4f}° · {start} → {end} · {int(st.session_state.radius_km)} km"
    st.markdown(loc_line)

    if st.button("Analizi yenile", type="primary", key="refresh_inference_btn"):
        st.session_state.inference_location_key = None
        st.rerun()

    inf = st.session_state.get("last_inference")
    if not inf:
        st.info("Sidebar’dan il ve tarih aralığını seçin; modeller otomatik çalışır.")
        return

    render_kpi_row(inf)

    col_a, col_b = st.columns(2)
    with col_a:
        render_rabia_card(inf.get("rabia", {}))
    with col_b:
        render_onur_card(inf.get("onur", {}))

    commentary = build_agent_commentary(
        inf,
        province,
        lat,
        lon,
        st.session_state.get("user_note", ""),
        f"{start} — {end}",
    )
    render_commentary_box(commentary)


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_global_styles()
    ui_sidebar()

    render_header()

    try:
        pred_df = load_predictions()
        events = load_raw_events()
    except Exception as exc:
        st.error(f"Veri yüklenemedi: {exc}")
        pred_df = pd.DataFrame()
        events = pd.DataFrame()

    tab_home, tab_map, tab_outputs, tab_live = st.tabs(
        ["Ana Sayfa", "Risk Haritası", "Model Çıktıları", "Canlı Çıkarım"]
    )

    with tab_home:
        try:
            _tab_home()
        except Exception as exc:
            st.error(f"Ana sayfa: {exc}")
            st.exception(exc)

    with tab_map:
        try:
            if pred_df.empty:
                st.warning("outputs/05_test_predictions.csv bulunamadı.")
            else:
                _tab_risk_map(pred_df)
        except Exception as exc:
            st.error(f"Harita: {exc}")
            st.exception(exc)

    with tab_outputs:
        try:
            _tab_model_outputs(pred_df, events)
        except Exception as exc:
            st.error(f"Model çıktıları: {exc}")
            st.exception(exc)

    with tab_live:
        try:
            _tab_live_inference()
        except Exception as exc:
            st.error(f"Canlı çıkarım: {exc}")
            st.exception(exc)

    render_footer(sidebar=False)


if __name__ == "__main__":
    main()
