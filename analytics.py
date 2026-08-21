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


def _is_bot_visit() -> bool:
    """人間の利用ではないアクセスかどうか。

    アプリを起こしておくため .github/workflows/keep-alive.yml が1日6回訪問する。
    これを数えると、本来知りたい「実際に使った人」がボットに埋もれてしまう。

    判定は2つ用意している。どちらか一方でも該当すれば計測しない:
      1. 自動アクセス側が付ける目印(?keepalive=1)
      2. ヘッドレスブラウザのUser-Agent
    目印だけに頼ると、Streamlitの配信構造やデプロイの反映遅れで取りこぼした際に
    「気づかないまま計測が汚れる」ため、実体(ヘッドレス)側からも見る。
    """
    try:
        if st.query_params.get("keepalive") == "1":
            return True
    except Exception:
        pass

    try:
        user_agent = st.context.headers.get("User-Agent", "")
    except Exception:
        return False

    # 人間のブラウザには出てこない文字列。keep-aliveのPlaywrightはこれに該当する
    return any(marker in user_agent for marker in ("HeadlessChrome", "Headless", "bot", "Bot"))


def track_event(name: str, params: dict | None = None) -> None:
    """GA4にイベントを1件送信する。失敗してもアプリの動作は止めない。"""
    if _is_bot_visit():
        return

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
