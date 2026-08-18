"""GA4 Measurement Protocol 経由でアプリの利用状況を送信する薄いラッパー。

Streamlitはサーバー側で完結するため、gtag.js のようなブラウザ計測が使えない
(st.components.v1.html はiframe内で動くため、ページ単位の計測とは相性が悪い)。
代わりにGA4のMeasurement Protocol(HTTPS POST)でイベントを直接送る。

測定ID・APIシークレットは st.secrets["ga"] から読む。未設定(ローカル開発など)なら
何もせず黙って抜ける — 計測の有無でアプリの動作が変わらないようにするため。
"""
import uuid

import requests
import streamlit as st

_ENDPOINT = "https://www.google-analytics.com/mp/collect"
_TIMEOUT_SECONDS = 2


def _get_client_id() -> str:
    if "_ga_client_id" not in st.session_state:
        st.session_state["_ga_client_id"] = str(uuid.uuid4())
    return st.session_state["_ga_client_id"]


def track_event(name: str, params: dict | None = None) -> None:
    """GA4にイベントを1件送信する。失敗してもアプリの動作は止めない。"""
    try:
        measurement_id = st.secrets["ga"]["measurement_id"]
        api_secret = st.secrets["ga"]["api_secret"]
    except Exception:
        return

    payload = {
        "client_id": _get_client_id(),
        "events": [{"name": name, "params": params or {}}],
    }
    try:
        requests.post(
            _ENDPOINT,
            params={"measurement_id": measurement_id, "api_secret": api_secret},
            json=payload,
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception:
        pass
