"""テスト共通の準備。

計算部分は core.py にまとまっているため、そのまま import して検証できる。
`app` フィクスチャは画面(app.py)側の関数も見たいとき用で、Streamlit を
ダミーに差し替えてから読み込む。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


class _DummyCtx:
    """with 文で使われる Streamlit の各種コンテナの代役。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _DummyStreamlit(types.ModuleType):
    """呼ばれても何もしないダミー。

    - 既定の戻り値は None(falsy)。`if st.button(...)` の分岐に入らないようにする
    - columns はコンテキストマネージャのリストを返す(`with col2:` を通すため)
    - stop は SystemExit を投げ、画面処理の入り口で実行を止める
    """

    _state: dict = {}

    def __getattr__(self, name):
        if name == "session_state":
            return _DummyStreamlit._state

        def _noop(*args, **kwargs):
            if name == "columns":
                spec = args[0] if args else 1
                n = len(spec) if isinstance(spec, (list, tuple)) else spec
                return [_DummyCtx() for _ in range(n)]
            if name in ("container", "expander", "form", "spinner", "sidebar", "tabs"):
                return _DummyCtx()
            if name == "stop":
                raise SystemExit("st.stop()")
            return None

        return _noop


def _load_app():
    """Streamlit をダミー化した状態で app.py を読み込む。

    app.py は途中で st.stop() に到達して実行が止まる。通常の import では
    例外時にモジュールが破棄されるため、手動で exec して
    「関数定義まで済んだ状態」を保持する。
    """
    sys.modules["streamlit"] = _DummyStreamlit("streamlit")
    spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["app"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass  # 画面処理の入り口で止まるのは想定どおり
    return module


@pytest.fixture(scope="session")
def core():
    """計算部分。Streamlit に依存しないので普通に import できる。"""
    sys.path.insert(0, str(ROOT))
    import core as core_module

    return core_module


@pytest.fixture(scope="session")
def app():
    """画面側。Streamlit をダミー化して読み込む。"""
    return _load_app()


@pytest.fixture(scope="session")
def orders_df():
    """構造(曜日・月・月末)がはっきりした日次受注データ。"""
    df = pd.read_csv(ROOT / "sample_data" / "sample_orders_daily.csv", encoding="utf-8-sig")
    df["日付"] = pd.to_datetime(df["日付"])
    df["受注件数"] = pd.to_numeric(df["受注件数"])
    return df


@pytest.fixture(scope="session")
def bento_df():
    """手がかり列つきのお弁当販売データ(要因分析モード用)。"""
    return pd.read_csv(ROOT / "sample_data" / "sample_bento_daily.csv", encoding="utf-8-sig")


@pytest.fixture(scope="session")
def small_df():
    """件数が少なく、学習が意図的に中断されるデータ。"""
    df = pd.read_csv(ROOT / "sample_data" / "sample_small_25rows.csv", encoding="utf-8-sig")
    df["日付"] = pd.to_datetime(df["日付"])
    df["売上"] = pd.to_numeric(df["売上"])
    return df


@pytest.fixture(scope="session")
def weekly_df():
    """週次(緩やかな上昇トレンドあり)の合成データ。"""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=104, freq="W-MON"),
        "sales": 200 + np.arange(104) * 1.5 + rng.normal(0, 15, 104),
    })


@pytest.fixture(scope="session")
def monthly_df():
    """月次(季節性あり)の合成データ。件数が少ない側の代表。"""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=36, freq="MS"),
        "sales": 1000
        + np.arange(36) * 10
        + 150 * np.sin(np.arange(36) / 12 * 2 * np.pi)
        + rng.normal(0, 40, 36),
    })
