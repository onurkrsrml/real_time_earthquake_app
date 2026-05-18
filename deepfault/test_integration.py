"""Entegrasyon testi."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ERRORS: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"OK  {name}")
    except Exception as exc:
        ERRORS.append(f"{name}: {exc}")
        print(f"FAIL {name}: {exc}")


def test_all() -> None:
    from deepfault.analytics import aggregate_predictions_near
    from deepfault.charts import fig_risk_timeseries, fig_top_risk_cells
    from deepfault.inference import load_predictions, load_raw_events, run_live_inference
    from deepfault.maps import build_risk_folium_map, build_risk_plotly_scatter_geo, prepare_map_data
    from deepfault.paths import LOGO_PATH

    assert LOGO_PATH.exists(), "Logo missing"

    pred = load_predictions()
    events = load_raw_events()
    from deepfault.data_bounds import prediction_date_bounds

    dmin, dmax = prediction_date_bounds()
    end = dmax
    start = max(dmin, end - timedelta(days=90))

    r1 = run_live_inference("İstanbul", 41.0, 29.0, 50, start, end)
    r2 = run_live_inference("Van", 38.5, 43.4, 120, start, end)
    check("scores differ by location", lambda: None if r1["combined_risk_score"] != r2["combined_risk_score"] else (_ for _ in ()).throw(ValueError("same")))

    r3 = run_live_inference("İstanbul", 41.0, 29.0, 50, end - timedelta(days=7), end)
    check("scores differ by date", lambda: None if r1["combined_risk_score"] != r3["combined_risk_score"] else (_ for _ in ()).throw(ValueError("same")))

    agg = aggregate_predictions_near(pred, 41.0, 29.0, 75, start, end)
    check("aggregate", lambda: None if not agg.get("empty") else (_ for _ in ()).throw(ValueError("empty")))

    map_df = prepare_map_data(pred, start, end)
    check("map", lambda: build_risk_plotly_scatter_geo(map_df))
    check(
        "folium map",
        lambda: build_risk_folium_map(
            map_df, center_lat=41, center_lon=29, highlight_lat=41, highlight_lon=29, province_label="IST", radius_km=75
        ),
    )

    if ERRORS:
        print("\nFAILED:", ERRORS)
        sys.exit(1)
    print("\nALL PASSED")


if __name__ == "__main__":
    test_all()
