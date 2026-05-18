"""Ajan için konum ve kullanıcı notuna göre özel yorum üretimi."""

from __future__ import annotations

from typing import Any


def build_agent_commentary(
    inference: dict[str, Any],
    province: str | None,
    lat: float,
    lon: float,
    user_note: str,
    date_range: str,
) -> str:
    lines: list[str] = []
    loc = province or "Özel koordinat"
    lines.append(f"📍 **Konum:** {loc} ({lat:.4f}°, {lon:.4f}°)")
    lines.append(f"📅 **Tarih aralığı:** {date_range}")

    rabia = inference.get("rabia", {})
    onur = inference.get("onur", {})
    agg = inference.get("province_aggregate")
    combined = inference.get("combined_risk_score")

    if combined is not None:
        lines.append(f"📊 **Birleşik risk skoru:** {combined:.1f} / 100")

    if rabia.get("ok"):
        lines.append(
            f"🧠 **Rabia XGBoost (kalibre):** 7 gün içinde M≥4 olasılığı **%{rabia['risk_probability']*100:.1f}** "
            f"({rabia['icon']} {rabia['risk_level']}). Eşik: {rabia['threshold']:.3f}."
        )
        if rabia.get("cell_id"):
            lines.append(f"   Grid hücresi: `{rabia['cell_id']}`")
    elif rabia.get("error"):
        lines.append(f"⚠️ Rabia modeli: {rabia['error']}")

    if onur.get("ok"):
        lines.append(
            f"🌋 **Onur regresyon:** Tahmini max büyüklük (7g) **{onur['pred_max_magnitude_7d']:.2f}**, "
            f"büyük olaya kalan gün **{onur['pred_days_to_major']:.1f}**, güven **{onur['confidence']:.2f}**."
        )
        if onur.get("reference_event_time"):
            lines.append(f"   Referans son olay: {onur['reference_event_time']}")
    elif onur.get("error"):
        lines.append(f"⚠️ Onur modeli: {onur['error']}")

    if agg and not agg.get("empty"):
        lines.append(
            f"🗺️ **Grid agregasyonu** ({agg['n_points']} nokta, {agg.get('date_start', '—')} → {agg.get('date_end', '—')}): "
            f"ortalama olasılık **%{agg['mean_probability']*100:.1f}**, "
            f"maks **%{agg['max_probability']*100:.1f}** (yarıçap: {inference.get('radius_km', '—')} km)."
        )

    lines.append("")
    lines.append("**Yorum:**")
    commentary = _interpret(inference, province, user_note)
    lines.append(commentary)

    if user_note and user_note.strip():
        lines.append("")
        lines.append(f"**Kullanıcı notu dikkate alındı:** {user_note.strip()}")

    lines.append("")
    lines.append(
        "_Bu çıktı istatistiksel risk değerlendirmesidir; resmi deprem erken uyarısı değildir._"
    )
    return "\n".join(lines)


def _interpret(inference: dict, province: str | None, user_note: str) -> str:
    rabia = inference.get("rabia", {})
    onur = inference.get("onur", {})
    combined = inference.get("combined_risk_score")
    parts: list[str] = []

    loc_phrase = f"{province} ve çevresi" if province else "seçilen koordinat çevresi"

    if combined is not None:
        if combined >= 65:
            parts.append(
                f"{loc_phrase} için kısa vadeli sismik aktivite artışı **belirgin** görünüyor; "
                "özellikle grid agregasyonu ve/veya sınıflandırma modeli uyumlu sinyal veriyor."
            )
        elif combined >= 40:
            parts.append(
                f"{loc_phrase} için risk **orta düzeyde**; izleme ve veri güncellemesi önerilir."
            )
        else:
            parts.append(
                f"{loc_phrase} için mevcut verilere göre risk **görece düşük**; "
                "yine de M≥4 olasılığı sıfır değildir."
            )

    if rabia.get("ok") and rabia["risk_probability"] >= 0.5:
        parts.append(
            "XGBoost sınıflandırıcı, önümüzdeki 7 gün içinde aynı grid hücresinde "
            "güçlü deprem olasılığını eşiğin üzerinde değerlendiriyor."
        )

    if onur.get("ok"):
        if onur["pred_max_magnitude_7d"] >= 4.5:
            parts.append(
                f"Regresyon modeli 7 günlük pencerede **{onur['pred_max_magnitude_7d']:.1f}** "
                "civarı maksimum büyüklük öngörüyor."
            )
        if onur["pred_days_to_major"] <= 14:
            parts.append(
                f"Bir sonraki büyük olay için tahmini süre **{onur['pred_days_to_major']:.0f} gün** "
                "— kısa vadeli dikkat gerektirebilir."
            )

    note = (user_note or "").lower()
    if note:
        if any(w in note for w in ("yüksek", "artış", "endişe", "risk")):
            parts.append(
                "Kullanıcı notundaki risk vurgusu, model çıktılarıyla birlikte değerlendirildi; "
                "saha doğrulaması ve resmi kurum verileriyle çapraz kontrol önerilir."
            )
        elif any(w in note for w in ("düşük", "sakin", "normal")):
            parts.append(
                "Kullanıcı notu daha sakin bir beklenti içeriyor; model çıktıları farklıysa "
                "veri gecikmesi veya uzamsal ölçek farkı olabilir."
            )
        else:
            parts.append(
                "Kullanıcı notu bağlamsal bilgi olarak eklendi; nicel skorlar modele dayanır."
            )

    if not parts:
        return "Yeterli model çıktısı üretilemedi; veri kapsamını veya konum seçimini kontrol edin."

    return " ".join(parts)
