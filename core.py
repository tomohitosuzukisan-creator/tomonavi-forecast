"""需要予測の計算部分。

画面(Streamlit)には依存しない。ここだけを読めば「何をどう計算しているか」が
分かるようにしてある。画面側は app.py にある。
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 日本語フォント設定
# Windows(ローカル)とLinux(Streamlit Community Cloud)の両方で日本語を表示するため、
# 利用可能なフォントを順に探す。Cloud側は packages.txt で fonts-ipafont-gothic を入れる。
JP_FONT_CANDIDATES = [
    "Meiryo",            # Windows
    "Yu Gothic",         # Windows
    "MS Gothic",         # Windows
    "Hiragino Sans",     # macOS
    "IPAexGothic",       # Linux (fonts-ipafont-gothic)
    "IPAPGothic",        # Linux
    "Noto Sans CJK JP",  # Linux
    "Noto Sans JP",      # Linux
    "TakaoPGothic",      # Linux
]

def setup_japanese_font() -> str | None:
    """使える日本語フォントを探して設定する。見つからなければ None を返す。"""
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return None

    for name in JP_FONT_CANDIDATES:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False  # マイナス記号の豆腐対策
            return name
    return None


ACTIVE_JP_FONT = setup_japanese_font()


MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_ROWS = 5000
MAX_COLS = 20
# 要因分析で一度に使える手がかりの数。多すぎると「どれが効いているか」が読み取れなくなる
MAX_FEATURE_COLS = 10

MAX_FORECAST_PERIODS = 14
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")

GRANULARITY_LABELS = {
    "auto": "自動判定",
    "daily": "日次",
    "weekly": "週次",
    "monthly": "月次",
}

GRANULARITY_CONFIG = {
    "daily": {
        "display": "日次",
        "freq": "D",
        "feature_cols": ["lag1", "lag7", "rolling_mean_7", "dayofweek", "month"],
        "min_rows": 20,
        # 予測値と比較する「直近の実績」をとる期間。
        # 1点だけだと曜日変動に、7日だと月末の駆け込みに振り回されるため、4週で均す
        "recent_window": 28,
        "recent_label": "直近4週の平均",
    },
    "weekly": {
        "display": "週次",
        "freq": "W-MON",
        "feature_cols": ["lag1", "lag4", "rolling_mean_4", "weekofyear", "month"],
        "min_rows": 16,
        "recent_window": 4,
        "recent_label": "直近4週の平均",
    },
    "monthly": {
        "display": "月次",
        "freq": "MS",
        "feature_cols": ["lag1", "lag3", "lag12", "rolling_mean_3", "month", "quarter"],
        "min_rows": 18,
        "recent_window": 3,
        "recent_label": "直近3か月の平均",
    },
}


def diagnose_time_series_data(
    ts_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    granularity: str = "daily",
) -> list[dict]:
    """アップロードデータの品質を診断し、ユーザー向けコメントを返す。"""
    messages = []

    if ts_df.empty or target_col not in ts_df.columns:
        return [{
            "level": "error",
            "title": "有効なデータがありません",
            "message": "日付列または目的変数列に有効な値がないため、予測できません。CSVの列設定を確認してください。",
        }]

    target = pd.to_numeric(ts_df[target_col], errors="coerce").dropna()
    row_count = len(target)

    if row_count == 0:
        return [{
            "level": "error",
            "title": "目的変数が数値として読み取れません",
            "message": "予測したい列に数値以外の値が多く含まれている可能性があります。売上や客数などの数値列を選んでください。",
        }]

    min_rows = GRANULARITY_CONFIG.get(granularity, GRANULARITY_CONFIG["daily"])["min_rows"]
    display_name = GRANULARITY_CONFIG.get(granularity, GRANULARITY_CONFIG["daily"])["display"]

    if row_count < min_rows:
        messages.append({
            "level": "error",
            "title": "データ件数が不足しています",
            "message": f"{display_name}の予測には、少なくとも{min_rows}件程度の有効データが必要です。まずは期間を増やしたCSVで試してください。",
        })
    elif row_count < 30:
        messages.append({
            "level": "warning",
            "title": "データ件数が少なめです",
            "message": "予測は実行できますが、件数が少ないため精度や手がかりの分析が安定しない可能性があります。",
        })

    unique_count = target.nunique()
    if unique_count < 5:
        messages.append({
            "level": "warning",
            "title": "目的変数の変動が少ないです",
            "message": "同じような値が多いため、AIが需要変動のパターンを学習しにくい状態です。手がかりの効き具合が読み取れない場合があります。",
        })

    std_value = float(target.std()) if row_count >= 2 else 0.0
    mean_value = float(target.mean()) if row_count > 0 else 0.0
    cv = std_value / abs(mean_value) if mean_value != 0 else 0.0

    if std_value == 0:
        messages.append({
            "level": "warning",
            "title": "目的変数がほぼ一定です",
            "message": "売上や客数がほぼ同じ値で並んでいるため、予測モデルがパターンを見つけられず、手がかりの影響を判定できません。",
        })
    elif cv < 0.03:
        messages.append({
            "level": "info",
            "title": "需要のばらつきが小さいです",
            "message": "需要が安定しているデータです。AI予測よりも移動平均などのシンプルな方法でも十分な可能性があります。",
        })

    zero_ratio = float((target == 0).mean())
    if zero_ratio > 0.3:
        messages.append({
            "level": "warning",
            "title": "ゼロ値が多く含まれています",
            "message": "売上ゼロ・客数ゼロの日が多い場合、休業日や欠測値が混ざっている可能性があります。休業日フラグなどを追加すると改善しやすくなります。",
        })

    if date_col in ts_df.columns:
        dates = pd.to_datetime(ts_df[date_col], errors="coerce").dropna().sort_values()
        if len(dates) >= 2:
            total_days = (dates.iloc[-1] - dates.iloc[0]).days
            if granularity == "daily" and total_days < 60:
                messages.append({
                    "level": "info",
                    "title": "期間がやや短いです",
                    "message": "日次予測では、できれば3か月以上のデータがあると曜日や月内傾向を捉えやすくなります。",
                })
            if granularity == "monthly" and row_count < 24:
                messages.append({
                    "level": "info",
                    "title": "月次の季節性判断には期間が短めです",
                    "message": "月次予測で前年同月の傾向を見るには、できれば24か月以上のデータがあると安定します。",
                })

    if not messages:
        messages.append({
            "level": "success",
            "title": "データ品質は概ね良好です",
            "message": "件数・変動ともに大きな問題は見られません。予測結果を確認してください。",
        })

    return messages


def normalize_target_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_granularity(date_series: pd.Series) -> str:
    """日付間隔から粒度を推定する。"""
    dates = pd.Series(pd.to_datetime(date_series, errors="coerce").dropna().sort_values().unique())
    if len(dates) < 3:
        return "daily"

    diffs = pd.Series(dates).diff().dropna().dt.total_seconds() / 86400.0
    if diffs.empty:
        return "daily"

    median_days = float(diffs.median())
    if median_days <= 3:
        return "daily"
    if median_days <= 15:
        return "weekly"
    return "monthly"


def aggregate_timeseries(df: pd.DataFrame, date_col: str, target_col: str, granularity: str) -> pd.DataFrame:
    """粒度に合わせて集計し、時系列として整える。"""
    work_df = df[[date_col, target_col]].copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
    work_df = work_df.dropna(subset=[date_col, target_col]).copy()

    if work_df.empty:
        return work_df

    if granularity == "daily":
        work_df[date_col] = work_df[date_col].dt.normalize()
    elif granularity == "weekly":
        # 週の開始日を月曜日に揃える
        work_df[date_col] = (work_df[date_col] - pd.to_timedelta(work_df[date_col].dt.dayofweek, unit="D")).dt.normalize()
    elif granularity == "monthly":
        work_df[date_col] = work_df[date_col].dt.to_period("M").dt.to_timestamp()

    work_df = (
        work_df.groupby(date_col, as_index=False)[target_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    return work_df


def add_time_features(df: pd.DataFrame, date_col: str, target_col: str, granularity: str) -> pd.DataFrame:
    """リークを避けるため、rolling は shift(1) の後に計算する。"""
    cfg = GRANULARITY_CONFIG[granularity]
    work_df = df.copy().sort_values(date_col).reset_index(drop=True)
    target_shifted = work_df[target_col].shift(1)

    if granularity == "daily":
        work_df["lag1"] = work_df[target_col].shift(1)
        work_df["lag7"] = work_df[target_col].shift(7)
        work_df["rolling_mean_7"] = target_shifted.rolling(7).mean()
        work_df["dayofweek"] = work_df[date_col].dt.dayofweek
        work_df["month"] = work_df[date_col].dt.month
    elif granularity == "weekly":
        work_df["lag1"] = work_df[target_col].shift(1)
        work_df["lag4"] = work_df[target_col].shift(4)
        work_df["rolling_mean_4"] = target_shifted.rolling(4).mean()
        work_df["weekofyear"] = work_df[date_col].dt.isocalendar().week.astype(int)
        work_df["month"] = work_df[date_col].dt.month
    elif granularity == "monthly":
        work_df["lag1"] = work_df[target_col].shift(1)
        work_df["lag3"] = work_df[target_col].shift(3)
        work_df["lag12"] = work_df[target_col].shift(12)
        work_df["rolling_mean_3"] = target_shifted.rolling(3).mean()
        work_df["month"] = work_df[date_col].dt.month
        work_df["quarter"] = work_df[date_col].dt.quarter

    return work_df


def build_next_date(last_date: pd.Timestamp, granularity: str) -> pd.Timestamp:
    if granularity == "daily":
        return last_date + pd.Timedelta(days=1)
    if granularity == "weekly":
        return last_date + pd.Timedelta(weeks=1)
    return last_date + pd.DateOffset(months=1)


def build_feature_row(history: pd.Series, next_date: pd.Timestamp, granularity: str) -> dict:
    values = history.astype(float).dropna().tolist()
    if not values:
        raise ValueError("予測に使える履歴データがありません。")

    def safe_tail_mean(window: int) -> float:
        tail = values[-window:] if len(values) >= window else values
        return float(np.mean(tail))

    row = {}
    if granularity == "daily":
        row["lag1"] = float(values[-1])
        row["lag7"] = float(values[-7]) if len(values) >= 7 else safe_tail_mean(7)
        row["rolling_mean_7"] = safe_tail_mean(7)
        row["dayofweek"] = int(next_date.dayofweek)
        row["month"] = int(next_date.month)
    elif granularity == "weekly":
        row["lag1"] = float(values[-1])
        row["lag4"] = float(values[-4]) if len(values) >= 4 else safe_tail_mean(4)
        row["rolling_mean_4"] = safe_tail_mean(4)
        row["weekofyear"] = int(next_date.isocalendar().week)
        row["month"] = int(next_date.month)
    elif granularity == "monthly":
        row["lag1"] = float(values[-1])
        row["lag3"] = float(values[-3]) if len(values) >= 3 else safe_tail_mean(3)
        row["lag12"] = float(values[-12]) if len(values) >= 12 else safe_tail_mean(12)
        row["rolling_mean_3"] = safe_tail_mean(3)
        row["month"] = int(next_date.month)
        row["quarter"] = int(next_date.quarter)

    return row


def calculate_metrics(y_true, y_pred):
    """MAE / RMSE / MAPE を計算する。MAPEは実績0を除外して算出する。"""
    y_true_array = pd.Series(y_true).astype(float).to_numpy()
    y_pred_array = pd.Series(y_pred).astype(float).to_numpy()

    mae = mean_absolute_error(y_true_array, y_pred_array)
    rmse = mean_squared_error(y_true_array, y_pred_array) ** 0.5

    non_zero_mask = y_true_array != 0
    if non_zero_mask.sum() == 0:
        mape = None
    else:
        mape = (
            abs(y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
            / abs(y_true_array[non_zero_mask])
        ).mean() * 100

    return mae, rmse, mape


def calculate_baseline_metrics(y_train, y_test, y_pred):
    """AIモデルと比較するための単純なベースライン精度を計算する。"""
    y_train_array = pd.Series(y_train).astype(float).to_numpy()
    y_test_array = pd.Series(y_test).astype(float).to_numpy()
    y_pred_array = pd.Series(y_pred).astype(float).to_numpy()

    if len(y_train_array) == 0 or len(y_test_array) == 0:
        return {
            "model_mape": None,
            "mean_mape": None,
            "naive_mape": None,
            "mean_improvement": None,
            "naive_improvement": None,
        }

    mean_pred = np.full(len(y_test_array), y_train_array.mean())
    naive_pred = np.full(len(y_test_array), y_train_array[-1])

    def calc_mape(y_true_array, pred_array):
        mask = y_true_array != 0
        if mask.sum() == 0:
            return None
        return (np.abs(y_true_array[mask] - pred_array[mask]) / np.abs(y_true_array[mask])).mean() * 100

    model_mape = calc_mape(y_test_array, y_pred_array)
    mean_mape = calc_mape(y_test_array, mean_pred)
    naive_mape = calc_mape(y_test_array, naive_pred)

    def improvement(base_mape, model_mape):
        if base_mape is None or model_mape is None or base_mape == 0:
            return None
        return (base_mape - model_mape) / base_mape * 100

    return {
        "model_mape": model_mape,
        "mean_mape": mean_mape,
        "naive_mape": naive_mape,
        "mean_improvement": improvement(mean_mape, model_mape),
        "naive_improvement": improvement(naive_mape, model_mape),
    }


def is_ai_lost(baseline_metrics: dict | None) -> bool:
    """AIが単純な方法(平均・直前値)のどちらかに負けたかどうか。判定不能ならFalse。"""
    bm = baseline_metrics or {}
    model_mape = bm.get("model_mape")
    mean_mape = bm.get("mean_mape")
    naive_mape = bm.get("naive_mape")
    if model_mape is None or mean_mape is None or naive_mape is None:
        return False
    # numpy の真偽値が漏れないよう、Python の bool に揃えて返す
    return bool(model_mape >= min(mean_mape, naive_mape))


def generate_forecast_summary(latest_actual: float, future_df: pd.DataFrame) -> dict:
    """予測結果を、業種を問わず使える見通しサマリーに変換する。"""
    empty_summary = {
        "trend": "判定不可",
        "trend_message": "予測値から十分な傾向を判定できませんでした。",
        "trend_level": "warning",
        "stability": "判定不可",
        "stability_message": "予測値のばらつきが確認できませんでした。",
        "avg_value": 0.0,
        "max_value": 0.0,
        "min_value": 0.0,
        "change_ratio": 0.0,
        "volatility_ratio": 0.0,
    }

    if future_df.empty or "予測値" not in future_df.columns:
        return empty_summary

    values = pd.to_numeric(future_df["予測値"], errors="coerce").dropna()
    if values.empty:
        return empty_summary

    avg_value = float(values.mean())
    max_value = float(values.max())
    min_value = float(values.min())
    std_value = float(values.std()) if len(values) >= 2 else 0.0

    change_ratio = (avg_value - latest_actual) / latest_actual if latest_actual != 0 else 0.0

    if change_ratio >= 0.20:
        trend = "大きく増加"
        trend_message = "直近の平均を大きく上回る見通しです。需要や件数が強く増える可能性があります。"
        trend_level = "success"
    elif change_ratio >= 0.10:
        trend = "増加傾向"
        trend_message = "直近の平均より高めの見通しです。需要や件数が増える可能性があります。"
        trend_level = "info"
    elif change_ratio <= -0.20:
        trend = "大きく減少"
        trend_message = "直近の平均を大きく下回る見通しです。需要や件数が強く減る可能性があります。"
        trend_level = "error"
    elif change_ratio <= -0.10:
        trend = "減少傾向"
        trend_message = "直近の平均より低めの見通しです。需要や件数が減る可能性があります。"
        trend_level = "warning"
    else:
        trend = "横ばい"
        trend_message = "直近の平均に近い水準で推移する可能性があります。"
        trend_level = "info"

    volatility_ratio = std_value / abs(avg_value) if avg_value != 0 else 0.0

    if volatility_ratio < 0.05:
        stability = "安定"
        stability_message = "予測期間内のばらつきは小さく、比較的読みやすい見通しです。"
    elif volatility_ratio < 0.15:
        stability = "やや変動あり"
        stability_message = "予測期間内で一定のばらつきがあります。ピークと谷の差を確認してください。"
    else:
        stability = "変動が大きい"
        stability_message = "予測期間内のばらつきが大きめです。平均値だけでなく最大・最小も確認してください。"

    return {
        "trend": trend,
        "trend_message": trend_message,
        "trend_level": trend_level,
        "stability": stability,
        "stability_message": stability_message,
        "avg_value": avg_value,
        "max_value": max_value,
        "min_value": min_value,
        "change_ratio": change_ratio,
        "volatility_ratio": volatility_ratio,
    }


def explain_feature_importance(importance_df: pd.DataFrame) -> list[str]:
    """特徴量重要度を、断定しすぎない日本語の解釈に変換する。"""
    if importance_df.empty or "重要度" not in importance_df.columns:
        return []

    work_df = importance_df.copy()
    work_df["重要度"] = pd.to_numeric(work_df["重要度"], errors="coerce").fillna(0)

    total_importance = work_df["重要度"].sum()
    if total_importance == 0:
        return [
            "・今回のデータでは、需要の変動要因を十分に判定できませんでした。",
            "※この解釈は統計的な傾向に基づく参考情報であり、因果関係を示すものではありません。",
        ]

    work_df["重要度割合"] = work_df["重要度"] / total_importance
    top_features = work_df.sort_values("重要度", ascending=False).head(3)
    messages = []

    feature_messages = {
        "lag1": "直前の実績",
        "lag3": "3期間前の実績",
        "lag4": "4期間前の実績",
        "lag7": "7日前の実績",
        "lag12": "12期間前の実績",
        "rolling_mean_3": "直近3期間の平均",
        "rolling_mean_4": "直近4期間の平均",
        "rolling_mean_7": "直近7日間の平均",
        "dayofweek": "曜日",
        "weekofyear": "年内の週番号",
        "month": "月",
        "quarter": "四半期",
    }

    detail_messages = {
        "lag1": "直近の増減が続きやすいデータである可能性があります。",
        "lag3": "短期的な周期性がある可能性があります。",
        "lag4": "週次データでは、約1か月前の動きが参考になっている可能性があります。",
        "lag7": "曜日ごとの繰り返し傾向がある可能性があります。",
        "lag12": "月次データでは、前年同月のような季節性がある可能性があります。",
        "rolling_mean_3": "短期的なならし値が参考になっている可能性があります。",
        "rolling_mean_4": "週次の基調が参考になっている可能性があります。",
        "rolling_mean_7": "日々の細かな変動より、直近の基調が参考になっている可能性があります。",
        "dayofweek": "曜日別の需要差がある可能性があります。",
        "weekofyear": "季節や時期による変動がある可能性があります。",
        "month": "季節性や月ごとの需要差がある可能性があります。",
        "quarter": "季節や期ごとの傾向がある可能性があります。",
    }

    for _, row in top_features.iterrows():
        feature = str(row["特徴量"])
        importance = float(row["重要度"])
        ratio = float(row["重要度割合"])

        if importance <= 0:
            continue

        if ratio >= 0.50:
            strength = "強く影響している可能性があります"
        elif ratio >= 0.20:
            strength = "一定の影響がある可能性があります"
        else:
            strength = "一部影響している可能性があります"

        feature_name = feature_messages.get(feature, feature)
        detail = detail_messages.get(
            feature,
            "この項目と予測対象の関係を確認すると、改善のヒントになる可能性があります。",
        )

        messages.append(
            f"・{feature_name}が予測に{strength}（重要度割合：約{ratio * 100:.0f}%）。{detail}"
        )

    messages.append("※この解釈は統計的な傾向に基づく参考情報であり、因果関係を示すものではありません。")
    return messages


def generate_comment(
    latest_actual: float, future_mean: float, recent_label: str = "直近実績"
) -> str:
    """直近実績と予測平均を比べた補足コメントを返す。

    設計方針として、断定や具体的な業務アクション(発注を増やす等)の指示は行わない。
    予測対象が売上・客数・生産量など多様なため、汎用的な事実の提示にとどめる。
    """
    if latest_actual == 0:
        return (
            f"{recent_label}が0のため増減率での比較ができません。"
            f"予測期間の平均は {future_mean:,.1f} です。"
            "実績に休業日や欠測が混ざっていないか確認してください。"
        )

    change_ratio = (future_mean - latest_actual) / latest_actual * 100

    if change_ratio >= 10:
        direction = (
            f"{recent_label} {latest_actual:,.1f} に対して、予測期間の平均は {future_mean:,.1f}"
            f"(約 {change_ratio:+.1f}%)と高めの水準です。"
        )
    elif change_ratio <= -10:
        direction = (
            f"{recent_label} {latest_actual:,.1f} に対して、予測期間の平均は {future_mean:,.1f}"
            f"(約 {change_ratio:+.1f}%)と低めの水準です。"
        )
    else:
        direction = (
            f"{recent_label} {latest_actual:,.1f} に対して、予測期間の平均は {future_mean:,.1f}"
            f"(約 {change_ratio:+.1f}%)とほぼ同水準です。"
        )

    return (
        direction
        + "この値は過去データのパターンから機械的に算出したものです。"
        "実際の判断では、すでに分かっている受注予定・販促・休業予定など、"
        "データに含まれていない情報も必ず併せて確認してください。"
    )


def make_time_series_forecast(
    model,
    history_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    granularity: str,
    forecast_periods: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    history = history_df[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)
    future_rows = []

    for step in range(1, forecast_periods + 1):
        next_date = build_next_date(history[date_col].iloc[-1], granularity)
        row = build_feature_row(history[target_col], next_date, granularity)
        row_df = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(row_df)[0])

        future_rows.append(
            {
                "予測期間": step,
                date_col: next_date,
                "予測値": pred,
            }
        )

        history = pd.concat(
            [history, pd.DataFrame({date_col: [next_date], target_col: [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)


def build_time_series_model(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    granularity: str,
    forecast_periods: int,
):
    ts_df = aggregate_timeseries(df, date_col, target_col, granularity)
    if ts_df.empty:
        raise ValueError("有効な日付と目的変数がありません。空のデータになっています。")

    feature_df = add_time_features(ts_df, date_col, target_col, granularity)
    feature_cols = GRANULARITY_CONFIG[granularity]["feature_cols"]
    model_df = feature_df.dropna(subset=feature_cols + [target_col]).copy()

    min_rows = GRANULARITY_CONFIG[granularity]["min_rows"]
    if len(model_df) < min_rows:
        raise ValueError(
            f"{GRANULARITY_CONFIG[granularity]['display']}の予測には、少なくとも {min_rows} 行以上の有効データが必要です。"
        )

    split_idx = max(int(len(model_df) * 0.8), min_rows - 2)
    if split_idx >= len(model_df):
        split_idx = len(model_df) - 1

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1,
        # 既定値(20)は中小企業の小規模データに厳しすぎ、月次などで分岐が一度も
        # 作られず重要度が全ゼロになる。小さな葉を許して解釈可能性を確保する
        min_child_samples=5,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae, rmse, mape = calculate_metrics(y_test, test_pred)
    baseline_metrics = calculate_baseline_metrics(y_train, y_test, test_pred)

    compare_df = pd.DataFrame(
        {
            "日付": test_df[date_col].values,
            "実績": y_test.values,
            "予測": test_pred,
        }
    )

    importance_df = pd.DataFrame(
        {
            "特徴量": feature_cols,
            "重要度": model.feature_importances_,
        }
    ).sort_values("重要度", ascending=False)

    future_df = make_time_series_forecast(
        model=model,
        history_df=model_df[[date_col, target_col]],
        date_col=date_col,
        target_col=target_col,
        granularity=granularity,
        forecast_periods=forecast_periods,
        feature_cols=feature_cols,
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "model_df": model_df,
        "compare_df": compare_df,
        "importance_df": importance_df,
        "future_df": future_df,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "baseline_metrics": baseline_metrics,
        "ts_df": ts_df,
    }


def diagnose_feature_columns(df: pd.DataFrame, date_col: str, target_col: str) -> list[dict]:
    """手がかり候補の各列を診断し、初期チェック状態と注意書きを決める。

    ID列・自由記述・目的変数のコピー(リーク)は、見かけの精度だけを
    水増しして解釈を壊すため、既定でチェックを外し理由を添える。
    """
    n = len(df)
    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    diagnoses = []
    for col in df.columns:
        if col in (date_col, target_col):
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(series)
        default_on = True
        note = ""

        if n > 0 and nunique >= n * 0.95:
            default_on = False
            note = (
                "ほぼ全行が異なる値です。通し番号やIDの可能性が高く、"
                "手がかりに入れると見かけの精度だけが良くなるため外してあります。"
            )
        elif not is_num and nunique > 50:
            default_on = False
            note = (
                f"種類が{nunique}もある文字列です。品番や自由記述の可能性が高く、"
                "パターンとして学習しにくいため外してあります。"
            )
        elif is_num:
            corr = target_numeric.corr(pd.to_numeric(series, errors="coerce"))
            if pd.notna(corr) and abs(float(corr)) > 0.98:
                default_on = False
                note = (
                    "予測したい数量とほぼ同じ動きをしています。数量のコピーや"
                    "派生列(リーク)の疑いがあるため外してあります。"
                )

        diagnoses.append({"col": col, "default_on": default_on, "note": note})
    return diagnoses


def explain_generic_importance(importance_df: pd.DataFrame, target_col: str) -> list[str]:
    """手がかりの効き具合を、断定しすぎない日本語にする(要因分析モード用)。"""
    total = float(importance_df["重要度"].sum())
    if total <= 0:
        return []
    lines = []
    for _, row in importance_df.head(3).iterrows():
        share = float(row["重要度"]) / total * 100
        if share < 10:
            continue
        lines.append(
            f"「{row['特徴量']}」が予測の約{share:.0f}%を担っています。"
            f"{row['特徴量']}によって{target_col}の水準が変わっている可能性があります。"
        )
    return lines


def build_additional_feature_model(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [date_col, target_col]]
    if len(feature_cols) == 0:
        raise ValueError("手がかりに使える列がありません。日付列と予測したい数量以外の列を用意してください。")

    model_df = df.copy()

    # 時系列分割の前提として、必ず日付順に並べ替える。
    # CSVの並び順のまま分割すると「未来のデータで学習して過去を当てる」ことが起こり得る
    model_df[date_col] = pd.to_datetime(model_df[date_col], errors="coerce")
    model_df = model_df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")

    numeric_features = []
    for col in feature_cols:
        # pandas 3.x では文字列列の dtype が object ではなく str になるため、
        # 「数値かどうか」で判定する(dtype=="object" 判定だとテキスト列が全てNaN化して全行消える)
        if pd.api.types.is_numeric_dtype(model_df[col]):
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            numeric_features.append(col)
        else:
            # cat.codes で擬似的な大小関係を作らず、LightGBMのカテゴリ型としてそのまま渡す
            model_df[col] = model_df[col].astype("category")

    selected_features = list(feature_cols)

    # 欠損で行を落とすのは数値の手がかりと目的変数のみ。カテゴリ列の欠損はLightGBMが欠損として扱える
    model_df = model_df.dropna(subset=numeric_features + [target_col]).copy()
    if len(model_df) < 20:
        raise ValueError("要因分析には、少なくとも 20 行以上の有効データが必要です。")

    split_idx = int(len(model_df) * 0.8)
    if split_idx <= 0 or split_idx >= len(model_df):
        raise ValueError("学習用データと検証用データを分けられませんでした。データ件数を確認してください。")

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train = train_df[selected_features]
    y_train = train_df[target_col]
    X_test = test_df[selected_features]
    y_test = test_df[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1,
        # 既定値は小規模データに厳しすぎ、カテゴリ列(メニュー等)の分岐が一度も
        # 作られず「効いていない」ように見えてしまう。小規模・少カテゴリ向けに緩める
        min_child_samples=5,
        min_data_per_group=10,
        cat_smooth=1.0,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae, rmse, mape = calculate_metrics(y_test, test_pred)
    baseline_metrics = calculate_baseline_metrics(y_train, y_test, test_pred)

    compare_df = pd.DataFrame(
        {
            "日付": test_df[date_col].values,
            "実績": y_test.values,
            "予測": test_pred,
        }
    )

    importance_df = pd.DataFrame(
        {
            "特徴量": selected_features,
            "重要度": model.feature_importances_,
        }
    ).sort_values("重要度", ascending=False)

    return {
        "model": model,
        "selected_features": selected_features,
        "compare_df": compare_df,
        "importance_df": importance_df,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "baseline_metrics": baseline_metrics,
    }
