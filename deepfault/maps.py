"""Risk haritası — gerçek tahmin verisi (outputs/05_test_predictions.csv)."""

from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import HeatMap

from deepfault.config import TURKEY_BOUNDS


def _prob_color(prob: float) -> str:
    """prob 0–1 aralığında."""
    pct = prob * 100
    if pct >= 60:
        return "#b71c1c"
    if pct >= 40:
        return "#e65100"
    if pct >= 25:
        return "#f9a825"
    return "#2e7d32"


def prepare_map_data(
    pred_df: pd.DataFrame,
    date_start=None,
    date_end=None,
    map_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Harita için hücre bazlı risk (tarih aralığında ortalama veya tek gün)."""
    from datetime import date as date_type

    from deepfault.analytics import filter_predictions_by_date, map_cells_in_range

    df = pred_df.dropna(subset=["latitude", "longitude", "risk_score"]).copy()
    if df.empty:
        return df

    if date_start is not None and date_end is not None:
        if not isinstance(date_start, date_type):
            date_start = pd.Timestamp(date_start).date()
        if not isinstance(date_end, date_type):
            date_end = pd.Timestamp(date_end).date()
        return map_cells_in_range(df, date_start, date_end)

    if map_date is not None and "date" in df.columns:
        slice_df = df[df["date"] == map_date]
        if not slice_df.empty:
            df = slice_df
    elif "date" in df.columns:
        latest = df["date"].max()
        df = df[df["date"] == latest]

    if "cell_id" in df.columns:
        df = df.sort_values("risk_score", ascending=False).drop_duplicates(
            subset=["cell_id"], keep="first"
        )
    return df


def build_risk_folium_map(
    map_df: pd.DataFrame,
    center_lat: float | None = None,
    center_lon: float | None = None,
    highlight_lat: float | None = None,
    highlight_lon: float | None = None,
    province_label: str | None = None,
    radius_km: float = 75,
) -> folium.Map:
    clat = center_lat or (TURKEY_BOUNDS["lat_min"] + TURKEY_BOUNDS["lat_max"]) / 2
    clon = center_lon or (TURKEY_BOUNDS["lon_min"] + TURKEY_BOUNDS["lon_max"]) / 2

    m = folium.Map(
        location=[clat, clon],
        zoom_start=6,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    if map_df.empty:
        folium.Marker(
            [clat, clon],
            popup="Tahmin verisi yok",
            icon=folium.Icon(color="gray", icon="info-sign"),
        ).add_to(m)
        return m

    heat = [
        [row["latitude"], row["longitude"], float(row["risk_score"])]
        for _, row in map_df.iterrows()
    ]
    HeatMap(heat, radius=22, blur=18, min_opacity=0.35, max_zoom=12).add_to(m)

    for _, row in map_df.iterrows():
        prob = float(row["risk_score"])
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 + prob * 12,
            color=_prob_color(prob),
            fill=True,
            fill_color=_prob_color(prob),
            fill_opacity=0.65,
            weight=1,
            popup=folium.Popup(
                f"<b>{row.get('cell_id', '—')}</b><br>"
                f"Risk olasılığı: {prob:.3f}<br>"
                f"Enlem: {row['latitude']:.3f}<br>"
                f"Boylam: {row['longitude']:.3f}",
                max_width=240,
            ),
        ).add_to(m)

    if highlight_lat is not None and highlight_lon is not None:
        folium.Marker(
            [highlight_lat, highlight_lon],
            popup=province_label or "Seçili bölge",
            icon=folium.Icon(color="blue", icon="star"),
        ).add_to(m)
        folium.Circle(
            location=[highlight_lat, highlight_lon],
            radius=float(radius_km) * 1000,
            color="#00B4D8",
            fill=True,
            fill_color="#00B4D8",
            fill_opacity=0.08,
            weight=2,
            dash_array="8",
        ).add_to(m)

    if not map_df.empty:
        m.fit_bounds(
            [[map_df["latitude"].min(), map_df["longitude"].min()],
             [map_df["latitude"].max(), map_df["longitude"].max()]],
            padding=(30, 30),
        )

    return m


def build_risk_plotly_scatter_geo(map_df: pd.DataFrame):
    """Plotly scatter_geo — grid hücre risk olasılıkları."""
    import plotly.express as px
    import plotly.graph_objects as go

    if map_df.empty:
        return None

    plot_df = map_df.copy()
    if "cell_id" not in plot_df.columns:
        plot_df["cell_id"] = [
            f"{row.latitude:.2f}_{row.longitude:.2f}" for row in plot_df.itertuples()
        ]
    plot_df["risk_pct"] = (plot_df["risk_score"] * 100).round(1)

    size_col = plot_df["risk_score"].clip(lower=0.01)
    fig = px.scatter_geo(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="risk_score",
        size=size_col,
        size_max=18,
        hover_data={
            "cell_id": True,
            "risk_pct": ":.1f",
            "risk_score": ":.3f",
            "latitude": ":.3f",
            "longitude": ":.3f",
        },
        color_continuous_scale="YlOrRd",
        range_color=[0, 1],
        title="scatter_geo — Grid Hücre Risk Olasılıkları",
    )
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.35)",
        showland=True,
        landcolor="rgba(30,40,55,0.9)",
        showocean=True,
        oceancolor="rgba(10,15,25,0.95)",
        lataxis_range=[35, 43],
        lonaxis_range=[25, 46],
        resolution=50,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9BA8BC"),
    )
    return fig


def build_risk_plotly_map(map_df: pd.DataFrame):
    """Geriye dönük uyumluluk."""
    return build_risk_plotly_scatter_geo(map_df)
