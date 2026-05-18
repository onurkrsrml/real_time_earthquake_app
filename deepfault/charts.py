"""Plotly grafikleri — yalnızca gerçek veri."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#00B4D8"
ORANGE = "#FF8500"
BLUE_DARK = "#0077B6"
BG = "rgba(0,0,0,0)"


def chart_has_data(fig_kind: str, data: pd.DataFrame | dict | None) -> bool:
    """İl/tarih seçimine bağlı grafikler için veri var mı?"""
    if fig_kind in ("risk_timeseries", "magnitude", "daily_events", "top_cells"):
        return isinstance(data, pd.DataFrame) and not data.empty
    if fig_kind == "metrics":
        return isinstance(data, dict) and bool(data)
    return False


def _layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#E8EDF4")),
        paper_bgcolor=BG,
        plot_bgcolor="rgba(15,22,35,0.6)",
        font=dict(family="Inter, system-ui, sans-serif", color="#9BA8BC"),
        margin=dict(l=48, r=24, t=52, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#C9D4E3")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    )


def fig_risk_timeseries(ts_df: pd.DataFrame) -> go.Figure:
    if ts_df.empty:
        fig = go.Figure()
        fig.update_layout(**_layout("Bölgesel risk zaman serisi — veri yok"))
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts_df["date"],
            y=ts_df["mean_risk"] * 100,
            mode="lines+markers",
            name="Ortalama risk (%)",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df["date"],
            y=ts_df["max_risk"] * 100,
            mode="lines",
            name="Maks risk (%)",
            line=dict(color=ORANGE, width=1.5, dash="dot"),
        )
    )
    fig.update_layout(**_layout("Bölgesel Risk Zaman Serisi (XGBoost Tahminleri)\n"))
    fig.update_yaxes(title="Risk (%)")
    return fig


def fig_magnitude_distribution(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        fig = go.Figure()
        fig.update_layout(**_layout("Büyüklük dağılımı — veri yok"))
        return fig

    fig = px.histogram(
        events,
        x="magnitude",
        nbins=28,
        color_discrete_sequence=[ORANGE],
        labels={"magnitude": "Büyüklük (M)", "count": "Frekans"},
    )
    fig.update_layout(**_layout("Deprem Büyüklük Dağılımı (USGS)"))
    return fig


def fig_top_risk_cells(cells_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if cells_df.empty:
        fig = go.Figure()
        fig.update_layout(**_layout("En yüksek riskli hücreler — veri yok"))
        return fig

    top = cells_df.nlargest(top_n, "risk_score").sort_values("risk_score", ascending=True)
    fig = px.bar(
        top,
        x="risk_score",
        y="cell_id",
        orientation="h",
        color="risk_score",
        color_continuous_scale=[[0, "#1B4965"], [0.5, ORANGE], [1, "#D00000"]],
        labels={"risk_score": "Ortalama olasılık", "cell_id": "Grid hücresi"},
    )
    fig.update_layout(**_layout(f"En Yüksek Riskli {top_n} Grid Hücresi"), coloraxis_showscale=False)
    return fig


def fig_model_metrics(metrics: dict) -> go.Figure:
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    labels = ["Gerçek: Yok", "Gerçek: Var"]
    pred_labels = ["Tahmin: Yok", "Tahmin: Var"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=pred_labels,
            y=labels,
            colorscale=[[0, "#0d1b2a"], [0.5, BLUE], [1, ORANGE]],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(**_layout("XGBoost — Confusion Matrix (Test)"))
    return fig


def fig_daily_events(events: pd.DataFrame) -> go.Figure:
    if events.empty or "time" not in events.columns:
        fig = go.Figure()
        fig.update_layout(**_layout("Günlük aktivite — veri yok"))
        return fig

    daily = (
        events.assign(day=events["time"].dt.date)
        .groupby("day")
        .size()
        .reset_index(name="count")
    )
    fig = px.area(
        daily,
        x="day",
        y="count",
        color_discrete_sequence=[BLUE],
        labels={"count": "Olay sayısı", "day": "Tarih"},
    )
    fig.update_layout(**_layout("Günlük Sismik Aktivite"))
    return fig


def fig_metrics_summary(metrics: dict) -> go.Figure:
    names = ["F1", "Precision", "Recall", "AUC-PR"]
    values = [
        metrics.get("f1", 0),
        metrics.get("precision", 0),
        metrics.get("recall", 0),
        metrics.get("average_precision_auc_pr", 0),
    ]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[BLUE, BLUE_DARK, ORANGE, "#FFB703"],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(**_layout("Model Performans Metrikleri"), yaxis_range=[0, 1.05])
    return fig
