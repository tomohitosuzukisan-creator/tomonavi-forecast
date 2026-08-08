"""app.py の中核ロジックを Streamlit なしで検証するスモークテスト。

app.py はモジュール直下に Streamlit の描画コードを持つため、
そのまま import すると画面処理が走ってしまう。
ここでは streamlit をダミーに差し替えてから import し、
純粋な計算関数(集計・特徴量・学習・予測・指標)だけを検証する。

実行: .venv/Scripts/python.exe smoke_test.py
"""
import sys
import types
from pathlib import Path

import pandas as pd


# --- streamlit をダミーに差し替える -------------------------------------
class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _DummyStreamlit(types.ModuleType):
    """呼ばれても何もしないダミー。

    ポイント:
    - 既定の戻り値は None(falsy)。これで `if st.button(...)` 等の分岐に入らない
    - columns はコンテキストマネージャのリストを返す(`with col2:` を通すため)
    - file_uploader は None を返し、直後の st.stop() で画面処理を打ち切る
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


sys.modules["streamlit"] = _DummyStreamlit("streamlit")

ROOT = Path(__file__).parent

# app.py は途中で st.stop() に到達して実行が止まる。
# 通常の import だと例外時にモジュールが破棄されるため、
# 手動で exec して「関数定義まで済んだ状態」を保持する。
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
app = importlib.util.module_from_spec(_spec)
sys.modules["app"] = app
try:
    _spec.loader.exec_module(app)
except SystemExit:
    pass  # 画面処理の入り口で止まるのは想定どおり
FAIL = []
PASS = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(label)
        print(f"  OK   {label} {detail}")
    else:
        FAIL.append(label)
        print(f"  NG   {label} {detail}")


def run_pipeline(csv_path: Path, date_col: str, target_col: str, label: str):
    print(f"\n[{label}] {csv_path.name}")
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col])

    granularity = app.infer_granularity(df[date_col])
    check(f"{label}: 粒度判定", granularity in ("daily", "weekly", "monthly"), f"→ {granularity}")

    ts = app.aggregate_timeseries(df, date_col, target_col, granularity)
    check(f"{label}: 集計", not ts.empty, f"→ {len(ts)}行")

    msgs = app.diagnose_time_series_data(ts, date_col, target_col, granularity)
    levels = [m["level"] for m in msgs]
    check(f"{label}: 品質診断", len(msgs) > 0, f"→ {levels}")

    try:
        result = app.build_time_series_model(
            df=df, date_col=date_col, target_col=target_col,
            granularity=granularity, forecast_periods=7,
        )
    except ValueError as e:
        print(f"  --   {label}: 学習は意図的に中断 → {e}")
        return None

    check(f"{label}: 未来予測", len(result["future_df"]) == 7, f"→ {len(result['future_df'])}期間")
    check(f"{label}: MAE算出", result["mae"] is not None, f"→ {result['mae']:.2f}")
    mape = result["mape"]
    check(f"{label}: MAPE算出", True, f"→ {'判定不可' if mape is None else f'{mape:.1f}%'}")

    bm = result["baseline_metrics"]
    check(f"{label}: ベースライン比較", "mean_mape" in bm,
          f"→ AI {bm['model_mape']:.1f}% / 平均 {bm['mean_mape']:.1f}% / 直前値 {bm['naive_mape']:.1f}%")

    imp_total = result["importance_df"]["重要度"].sum()
    check(f"{label}: 特徴量重要度", True, f"→ 合計 {imp_total}")

    # 致命バグだった generate_comment が実際に動くか
    # 比較基準はアプリ本体と同じく「直近1周期の平均」にそろえる
    cfg = app.GRANULARITY_CONFIG[granularity]
    latest = float(result["ts_df"][target_col].tail(cfg["recent_window"]).mean())
    fmean = float(result["future_df"]["予測値"].mean())
    comment = app.generate_comment(latest, fmean, cfg["recent_label"])
    check(f"{label}: generate_comment", isinstance(comment, str) and len(comment) > 10,
          f"→ {comment[:44]}...")

    # 基準が最終1点だった頃は、曜日変動だけで「大きく減少」と誤警告していた
    ratio = (fmean - latest) / latest if latest else 0.0
    check(f"{label}: 増減率が過大でない", abs(ratio) < 0.20, f"→ {ratio * 100:+.1f}%")

    # CSVダウンロード用の変換(バグでここまで到達できていなかった)
    export = result["future_df"].copy()
    csv_data = export.to_csv(index=False, encoding="utf-8-sig")
    check(f"{label}: CSV書き出し", len(csv_data) > 0, f"→ {len(csv_data)}バイト")

    return result


print("=" * 60)
print("需要予測アプリ スモークテスト")
print("=" * 60)

print(f"\n日本語フォント: {app.ACTIVE_JP_FONT or '見つかりません(グラフが豆腐になります)'}")

r_orders = run_pipeline(ROOT / "sample_data/sample_orders_daily.csv", "日付", "受注件数", "受注546日")
run_pipeline(ROOT / "sample_data/sample_small_25rows.csv", "日付", "売上", "25行(データ不足)")

# サンプルデータの性格が意図どおりかを確認する
print("\n[サンプルデータの性格チェック]")
if r_orders is not None:
    bm = r_orders["baseline_metrics"]
    check("受注データ: AIが単純平均に勝つ", bm["model_mape"] < bm["mean_mape"],
          f"→ AI {bm['model_mape']:.1f}% < 平均 {bm['mean_mape']:.1f}%")
    check("受注データ: AIが直前値に勝つ", bm["model_mape"] < bm["naive_mape"],
          f"→ AI {bm['model_mape']:.1f}% < 直前値 {bm['naive_mape']:.1f}%")
    check("受注データ: 特徴量重要度が出る", r_orders["importance_df"]["重要度"].sum() > 0)


# 週次・月次データを合成して検証(引き継ぎ資料で「検証不足」とされていた部分)
import numpy as np  # noqa: E402

rng = np.random.default_rng(42)
weekly = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=104, freq="W-MON"),
    "sales": 200 + np.arange(104) * 1.5 + rng.normal(0, 15, 104),
})
weekly.to_csv(ROOT / "sample_data/_tmp_weekly.csv", index=False)
run_pipeline(ROOT / "sample_data/_tmp_weekly.csv", "date", "sales", "週次104週")

monthly = pd.DataFrame({
    "date": pd.date_range("2022-01-01", periods=36, freq="MS"),
    "sales": 1000 + np.arange(36) * 10
    + 150 * np.sin(np.arange(36) / 12 * 2 * np.pi) + rng.normal(0, 40, 36),
})
monthly.to_csv(ROOT / "sample_data/_tmp_monthly.csv", index=False)
run_pipeline(ROOT / "sample_data/_tmp_monthly.csv", "date", "sales", "月次36か月")

print("\n" + "=" * 60)
print(f"結果: 成功 {len(PASS)} / 失敗 {len(FAIL)}")
if FAIL:
    print("失敗した項目:")
    for f in FAIL:
        print(" -", f)
    sys.exit(1)
print("すべて成功しました。")
