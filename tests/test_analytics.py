"""計測(analytics.py)の検証。

いちばん守りたいのは「休止対策の自動アクセスを利用者として数えない」こと。
ここが壊れると、1日6回の巡回がそのまま利用実績に化けて、実際に使った人が
埋もれてしまう(2026-08-21に実際に混入していたのを修正した)。

判定は「目印(?keepalive=1)」と「ヘッドレスのUser-Agent」の二重にしてある。
目印だけでは、配信構造やデプロイ反映の都合で取りこぼしたときに気づけないため。
"""
import sys
import types

import pytest

_HUMAN_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)
_HEADLESS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) HeadlessChrome/139.0.0.0 Safari/537.36"
)


class _DummyContext:
    def __init__(self, user_agent):
        self.headers = {"User-Agent": user_agent}


class _DummyStreamlit(types.ModuleType):
    """analytics.py が触る範囲だけを持つ Streamlit の代役。"""

    def __init__(self, name, query_params, user_agent):
        super().__init__(name)
        self.session_state = {}
        self.query_params = query_params
        self.context = _DummyContext(user_agent)
        self.secrets = {"ga": {"measurement_id": "G-TEST", "api_secret": "dummy"}}


@pytest.fixture
def load_analytics(monkeypatch):
    """訪問の条件を差し替えた状態で analytics.py を読み込む。"""

    def _load(query_params=None, user_agent=_HUMAN_UA):
        dummy = _DummyStreamlit("streamlit", query_params or {}, user_agent)
        monkeypatch.setitem(sys.modules, "streamlit", dummy)
        sys.modules.pop("analytics", None)
        import analytics

        return analytics

    return _load


def _record_posts(monkeypatch, analytics):
    """送信内容を捕まえる。実際のネットワークアクセスはさせない。"""
    sent = []
    monkeypatch.setattr(
        analytics.requests, "post", lambda *a, **kw: sent.append(kw) or None
    )
    return sent


def test_目印つきのアクセスは計測しない(load_analytics, monkeypatch):
    analytics = load_analytics(query_params={"keepalive": "1"})
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")
    analytics.track_event("app_forecast_run", {"mode": "time_series"})

    assert sent == [], "keep-aliveの訪問がGA4に送られている"


def test_目印がなくてもヘッドレスなら計測しない(load_analytics, monkeypatch):
    # 目印を取りこぼしても、実体がヘッドレスなら計測しない(二重の防波堤)
    analytics = load_analytics(query_params={}, user_agent=_HEADLESS_UA)
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")

    assert sent == [], "ヘッドレスブラウザの訪問がGA4に送られている"


def test_通常のアクセスは計測する(load_analytics, monkeypatch):
    analytics = load_analytics()
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")

    assert len(sent) == 1, "通常の訪問が計測されていない"
    assert sent[0]["json"]["events"][0]["name"] == "app_view"


def test_目印が別の値なら計測する(load_analytics, monkeypatch):
    # ?keepalive=1 以外は利用者のアクセスとして扱う(誤って計測を止めないため)
    analytics = load_analytics(query_params={"keepalive": "0"})
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")

    assert len(sent) == 1


def test_判定が使えない環境でも計測は止まらない(load_analytics, monkeypatch):
    # query_params や context が無い環境で計測が全部落ちると、
    # 「誰も使っていない」と誤読してしまう。判定できないときは計測する側に倒す
    analytics = load_analytics()
    monkeypatch.delattr(analytics.st, "context", raising=False)
    monkeypatch.delattr(analytics.st, "query_params", raising=False)
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")

    assert len(sent) == 1
