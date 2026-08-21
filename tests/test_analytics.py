"""計測(analytics.py)の検証。

いちばん守りたいのは「休止対策の自動アクセスを利用者として数えない」こと。
ここが壊れると、1日6回の巡回がそのまま利用実績に化けて、実際に使った人が
埋もれてしまう(2026-08-21に実際に混入していたのを修正した)。
"""
import sys
import types

import pytest


class _DummyStreamlit(types.ModuleType):
    """analytics.py が触る範囲だけを持つ Streamlit の代役。"""

    def __init__(self, name, query_params=None):
        super().__init__(name)
        self.session_state = {}
        self.query_params = query_params if query_params is not None else {}
        self.secrets = {"ga": {"measurement_id": "G-TEST", "api_secret": "dummy"}}


@pytest.fixture
def load_analytics(monkeypatch):
    """query_params を差し替えた状態で analytics.py を読み込む。"""

    def _load(query_params):
        dummy = _DummyStreamlit("streamlit", query_params=query_params)
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


def test_自動巡回のアクセスは計測しない(load_analytics, monkeypatch):
    analytics = load_analytics({"keepalive": "1"})
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")
    analytics.track_event("app_forecast_run", {"mode": "time_series"})

    assert sent == [], "keep-aliveの訪問がGA4に送られている"


def test_通常のアクセスは計測する(load_analytics, monkeypatch):
    analytics = load_analytics({})
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")

    assert len(sent) == 1, "通常の訪問が計測されていない"
    assert sent[0]["json"]["events"][0]["name"] == "app_view"


def test_目印が別の値なら計測する(load_analytics, monkeypatch):
    # ?keepalive=1 以外は利用者のアクセスとして扱う(誤って計測を止めないため)
    analytics = load_analytics({"keepalive": "0"})
    sent = _record_posts(monkeypatch, analytics)

    analytics.track_event("app_view")

    assert len(sent) == 1
