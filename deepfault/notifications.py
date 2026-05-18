"""Webhook ve Telegram bildirim servisi."""

from __future__ import annotations

from typing import Any

import requests


def build_agent_payload(
    risk_score: float | None,
    selected_city_or_coordinate: str,
    date_range: str,
    model_output_summary: str,
    user_note: str,
) -> dict[str, Any]:
    return {
        "risk_score": risk_score,
        "selected_city_or_coordinate": selected_city_or_coordinate,
        "date_range": date_range,
        "model_output_summary": model_output_summary,
        "user_note": user_note,
    }


def send_webhook(url: str, payload: dict[str, Any], timeout: int = 15) -> tuple[bool, str]:
    if not url or not url.strip():
        return False, "Webhook URL boş."
    try:
        response = requests.post(url.strip(), json=payload, timeout=timeout)
        if response.ok:
            return True, f"Webhook başarılı (HTTP {response.status_code})."
        return False, f"Webhook hata: HTTP {response.status_code} — {response.text[:200]}"
    except requests.exceptions.Timeout:
        return False, "Webhook zaman aşımına uğradı."
    except requests.exceptions.RequestException as exc:
        return False, f"Webhook isteği başarısız: {exc}"


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    timeout: int = 15,
) -> tuple[bool, str]:
    if not bot_token.strip() or not chat_id.strip():
        return False, "Telegram Bot Token veya Chat ID eksik."
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id.strip(), "text": text[:4000]},
            timeout=timeout,
        )
        data = response.json()
        if response.ok and data.get("ok"):
            return True, "Telegram mesajı gönderildi."
        desc = data.get("description", response.text[:200])
        return False, f"Telegram hata: {desc}"
    except requests.exceptions.Timeout:
        return False, "Telegram zaman aşımına uğradı."
    except requests.exceptions.RequestException as exc:
        return False, f"Telegram isteği başarısız: {exc}"


def format_telegram_report(payload: dict[str, Any]) -> str:
    summary = str(payload.get("model_output_summary", "—"))
    summary = summary.replace("**", "").replace("*", "")
    note = payload.get("user_note") or "—"
    return (
        "DeepFault — Risk Raporu\n"
        f"Konum: {payload.get('selected_city_or_coordinate', '—')}\n"
        f"Risk Skoru: {payload.get('risk_score', '—')}\n"
        f"Tarih Aralığı: {payload.get('date_range', '—')}\n"
        f"Özet:\n{summary}\n"
        f"Kullanıcı Notu: {note}"
    )


def send_webhook_telegram(
    webhook_url: str,
    telegram_token: str,
    telegram_chat_id: str,
    payload: dict[str, Any],
) -> list[tuple[str, bool, str]]:
    """Webhook ve isteğe bağlı Telegram gönderimi; sonuç listesi döner."""
    results: list[tuple[str, bool, str]] = []

    ok, msg = send_webhook(webhook_url, payload)
    results.append(("Webhook", ok, msg))

    if telegram_token.strip() and telegram_chat_id.strip():
        report = format_telegram_report(payload)
        tok, tmsg = send_telegram_message(telegram_token, telegram_chat_id, report)
        results.append(("Telegram", tok, tmsg))

    return results
