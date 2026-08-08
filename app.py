import io

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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

st.set_page_config(page_title="需要予測アプリ", layout="wide")


MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_ROWS = 5000
MAX_COLS = 20
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
            "message": "予測は実行できますが、件数が少ないため精度や特徴量重要度が安定しない可能性があります。",
        })

    unique_count = target.nunique()
    if unique_count < 5:
        messages.append({
            "level": "warning",
            "title": "目的変数の変動が少ないです",
            "message": "同じような値が多いため、AIが需要変動のパターンを学習しにくい状態です。特徴量重要度が0になる場合があります。",
        })

    std_value = float(target.std()) if row_count >= 2 else 0.0
    mean_value = float(target.mean()) if row_count > 0 else 0.0
    cv = std_value / abs(mean_value) if mean_value != 0 else 0.0

    if std_value == 0:
        messages.append({
            "level": "warning",
            "title": "目的変数がほぼ一定です",
            "message": "売上や客数がほぼ同じ値で並んでいるため、予測モデルが分岐を作れず、特徴量の影響を判定できません。",
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
            "message": "件数・変動ともに大きな問題は見られません。予測結果と特徴量重要度を確認してください。",
        })

    return messages


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


def show_user_error(message: str) -> None:
    st.error(message)


def read_csv_with_fallbacks(uploaded_file) -> pd.DataFrame:
    """複数の文字コードを順番に試してCSVを読む。"""
    raw_bytes = uploaded_file.getvalue()

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except Exception:
            continue

    raise ValueError(
        "CSVを読み込めませんでした。文字コードを utf-8-sig、utf-8、cp932、shift_jis の順で試しましたが失敗しました。"
    )


def validate_uploaded_file(uploaded_file) -> None:
    if uploaded_file is None:
        return

    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError("ファイルサイズが大きすぎます。5MB以下のCSVをアップロードしてください。")


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


def render_accuracy_summary(
    mae: float,
    rmse: float,
    mape: float | None,
    future_df: pd.DataFrame | None = None,
    baseline_metrics: dict | None = None,
) -> None:
    """精度指標を、一般ユーザーにも分かりやすく実務視点で表示する。"""
    st.subheader("精度評価（実務視点）")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("MAPE", "判定不可" if mape is None else f"{mape:.1f}%")

    st.caption(
        "MAEは平均的な誤差、RMSEは大きな誤差をより重く見る指標、MAPEは平均して何％ズレたかを見る指標です。"
    )

    if mape is None:
        st.warning("実績値が0のみのため、MAPEによる精度評価はできません。MAEやRMSEを参考にしてください。")
    else:
        if mape < 10:
            st.success(
                f"過去データ上では、平均して±{mape:.1f}%程度のズレでした。比較的小さめの誤差ですが、業務で使えるかは許容できるズレ幅と比較してください。"
            )
        elif mape < 20:
            st.info(
                f"過去データ上では、平均して±{mape:.1f}%程度のズレでした。大まかな見通し把握には使えますが、細かな数量判断では注意が必要です。"
            )
        elif mape < 30:
            st.warning(
                f"過去データ上では、平均して±{mape:.1f}%程度のズレでした。傾向把握の参考として使い、重要な判断では補足情報も確認してください。"
            )
        else:
            st.error(
                f"過去データ上では、平均して±{mape:.1f}%程度のズレでした。データ件数・特徴量・外れ値などを見直すことをおすすめします。"
            )

    if future_df is not None and mape is not None and "予測値" in future_df.columns:
        future_values = pd.to_numeric(future_df["予測値"], errors="coerce").dropna()
        if not future_values.empty:
            avg_forecast = float(future_values.mean())
            error_amount = avg_forecast * (mape / 100)
            st.info(
                f"今回の予測平均 {avg_forecast:,.1f} に対して、目安として±{error_amount:,.1f} 程度のズレが発生する可能性があります。"
            )

    if baseline_metrics is not None:
        st.markdown("**単純な予測方法との比較**")

        b1, b2, b3 = st.columns(3)
        b1.metric("AIモデル MAPE", "判定不可" if baseline_metrics.get("model_mape") is None else f"{baseline_metrics['model_mape']:.1f}%")
        b2.metric("単純平均 MAPE", "判定不可" if baseline_metrics.get("mean_mape") is None else f"{baseline_metrics['mean_mape']:.1f}%")
        b3.metric("直前値 MAPE", "判定不可" if baseline_metrics.get("naive_mape") is None else f"{baseline_metrics['naive_mape']:.1f}%")

        mean_improvement = baseline_metrics.get("mean_improvement")
        naive_improvement = baseline_metrics.get("naive_improvement")

        if mean_improvement is not None:
            if mean_improvement > 5:
                st.success(f"AIモデルは、単純平均予測より約{mean_improvement:.1f}%誤差を改善しています。")
            elif mean_improvement >= -5:
                st.warning("AIモデルは、単純平均予測とほぼ同程度です。AIを使うメリットは限定的な可能性があります。")
            else:
                st.error(f"AIモデルは、単純平均予測より誤差が大きくなっています。データや特徴量の見直しをおすすめします。")

        if naive_improvement is not None:
            if naive_improvement > 5:
                st.success(f"AIモデルは、直前値をそのまま使う予測より約{naive_improvement:.1f}%誤差を改善しています。")
            elif naive_improvement >= -5:
                st.warning("AIモデルは、直前値をそのまま使う予測とほぼ同程度です。単純な方法でも十分な可能性があります。")
            else:
                st.error(f"AIモデルは、直前値をそのまま使う予測より誤差が大きくなっています。データや特徴量の見直しをおすすめします。")

        st.caption("※業務で使えるかどうかは、最終的には許容できるズレ幅、欠品・廃棄・機会損失などの影響額と比較して判断してください。")

def render_data_diagnosis(messages: list[dict]) -> None:
    """データ診断コメントをUIに表示する。"""
    st.subheader("データ品質診断")
    for item in messages:
        text = f"**{item['title']}**：{item['message']}"
        if item["level"] == "error":
            st.error(text)
        elif item["level"] == "warning":
            st.warning(text)
        elif item["level"] == "success":
            st.success(text)
        else:
            st.info(text)


def render_importance_diagnosis(importance_df: pd.DataFrame) -> None:
    """特徴量重要度が出ない場合に理由を補足する。"""
    if importance_df.empty or "重要度" not in importance_df.columns:
        return

    total_importance = importance_df["重要度"].sum()
    if total_importance == 0:
        st.warning(
            "特徴量重要度がすべて0です。これはバグとは限らず、データ件数が少ない、目的変数の変動が小さい、"
            "または特徴量から需要の違いを十分に説明できない場合に起こります。"
        )
        st.info(
            "改善のヒント：期間を長くする、曜日・天候・イベント・価格・販促有無など、需要が変わる理由になりそうな列を追加すると改善しやすくなります。"
        )


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


def render_forecast_summary(
    summary: dict,
    target_col: str,
    importance_df: pd.DataFrame | None = None,
    recent_label: str = "直近実績",
) -> None:
    """今後の見通しサマリーをUIに表示する。"""
    st.subheader("今後の見通しサマリー")
    st.caption("業種や用途を限定せず、予測結果を判断材料として整理しています。")

    c1, c2, c3 = st.columns(3)
    c1.metric("トレンド", summary["trend"])
    c2.metric("安定性", summary["stability"])
    c3.metric(f"{recent_label}との差", f"{summary['change_ratio'] * 100:+.1f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric(f"予測平均（{target_col}）", f"{summary['avg_value']:,.1f}")
    c5.metric(f"予測最大（{target_col}）", f"{summary['max_value']:,.1f}")
    c6.metric(f"予測最小（{target_col}）", f"{summary['min_value']:,.1f}")

    if summary["trend_level"] == "success":
        st.success(f"見通し：{summary['trend_message']}")
    elif summary["trend_level"] == "error":
        st.error(f"見通し：{summary['trend_message']}")
    elif summary["trend_level"] == "warning":
        st.warning(f"見通し：{summary['trend_message']}")
    else:
        st.info(f"見通し：{summary['trend_message']}")

    st.info(f"安定性：{summary['stability_message']}")

    if importance_df is not None:
        explanations = explain_feature_importance(importance_df)
        if explanations:
            st.markdown("**予測に影響している主な要因**")
            for message in explanations:
                st.write(message)


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


def build_additional_feature_model(df: pd.DataFrame, date_col: str, target_col: str):
    candidate_features = [col for col in df.columns if col not in [date_col, target_col]]
    if len(candidate_features) == 0:
        raise ValueError("追加特徴量として使える列がありません。日付列と目的変数以外の列を用意してください。")

    model_df = df.copy()
    selected_features = []
    for col in candidate_features:
        if model_df[col].dtype == "object":
            model_df[col] = model_df[col].astype("category").cat.codes
        else:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        selected_features.append(col)

    model_df = model_df.dropna(subset=selected_features + [target_col]).copy()
    if len(model_df) < 20:
        raise ValueError("追加特徴量での予測には、少なくとも 20 行以上の有効データが必要です。")

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


st.title("需要予測アプリ")
st.write("CSVファイルをアップロードして、日次・週次・月次の需要予測を行います。")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("最初からやり直す"):
        st.session_state["uploader_key"] += 1
        st.rerun()

st.info(
    "CSVの日付列と数量列を選ぶだけで予測できます。粒度(日次/週次/月次)は自動判定します。アップロードされたデータは保存しません。"
)

mode = st.radio(
    "使い方を選んでください",
    ["時系列予測", "追加特徴量で予測"],
    horizontal=True,
)

uploaded_file = st.file_uploader(
    "CSVファイルをアップロードしてください",
    type=["csv"],
    key=f"file_uploader_{st.session_state['uploader_key']}",
)


if uploaded_file is None:
    st.info("まずはCSVファイルをアップロードしてください。")
    st.stop()


try:
    validate_uploaded_file(uploaded_file)
    df = read_csv_with_fallbacks(uploaded_file)
except Exception as exc:
    show_user_error(str(exc))
    st.stop()


if df.empty:
    show_user_error("CSVにデータが入っていません。行があるCSVをアップロードしてください。")
    st.stop()

if len(df) > MAX_ROWS:
    show_user_error(f"行数が多すぎます。{MAX_ROWS} 行以下のCSVをアップロードしてください。")
    st.stop()

if len(df.columns) > MAX_COLS:
    show_user_error(f"列数が多すぎます。{MAX_COLS} 列以下のCSVをアップロードしてください。")
    st.stop()

st.subheader("アップロードされたデータ")
st.dataframe(df, use_container_width=True)

all_columns = df.columns.tolist()
if len(all_columns) < 2:
    show_user_error("少なくとも日付列と目的変数列の 2 列が必要です。")
    st.stop()

st.subheader("設定")
setting_col1, setting_col2, setting_col3 = st.columns(3)

with setting_col1:
    date_col = st.selectbox("日付列を選択", all_columns)

remaining_target_columns = [col for col in all_columns if col != date_col]
with setting_col2:
    target_col = st.selectbox("目的変数列を選択", remaining_target_columns)

with setting_col3:
    granularity_label = st.selectbox(
        "データ粒度を選択",
        ["自動判定", "日次", "週次", "月次"],
        index=0,
    )

granularity_key = {v: k for k, v in GRANULARITY_LABELS.items()}[granularity_label]

if mode == "時系列予測":
    try:
        df_time = df[[date_col, target_col]].copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
        df_time[target_col] = pd.to_numeric(df_time[target_col], errors="coerce")
        df_time = df_time.dropna(subset=[date_col, target_col]).copy()

        if df_time.empty:
            raise ValueError("日付列または目的変数列に有効な値がありません。")

        if granularity_key == "auto":
            detected_granularity = infer_granularity(df_time[date_col])
        else:
            detected_granularity = granularity_key

        config = GRANULARITY_CONFIG[detected_granularity]
        st.caption(f"判定結果: {config['display']} で処理します。")

        if len(df_time) < config["min_rows"]:
            raise ValueError(
                f"{config['display']}の予測には、少なくとも {config['min_rows']} 行以上のデータが必要です。"
            )

        forecast_periods = st.slider(
            "予測期間（何期間先）",
            min_value=1,
            max_value=MAX_FORECAST_PERIODS,
            value=min(7, MAX_FORECAST_PERIODS),
        )

        st.subheader("データの確認")
        c1, c2, c3 = st.columns(3)
        c1.metric("行数", f"{len(df_time):,}")
        c2.metric("目的変数合計", f"{df_time[target_col].sum():,.2f}")
        c3.metric("目的変数平均", f"{df_time[target_col].mean():,.2f}")

        preview_df = aggregate_timeseries(df_time, date_col, target_col, detected_granularity)
        st.dataframe(preview_df.head(20), use_container_width=True)

        diagnosis_messages = diagnose_time_series_data(
            preview_df,
            date_col,
            target_col,
            detected_granularity,
        )
        render_data_diagnosis(diagnosis_messages)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(preview_df[date_col], preview_df[target_col], marker="o")
        ax.set_xlabel("日付")
        ax.set_ylabel(target_col)
        ax.set_title(f"{target_col} の推移")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        if st.button("時系列モデルで予測する"):
            result = build_time_series_model(
                df=df_time,
                date_col=date_col,
                target_col=target_col,
                granularity=detected_granularity,
                forecast_periods=forecast_periods,
            )

            render_accuracy_summary(
                result["mae"],
                result["rmse"],
                result["mape"],
                result["future_df"],
                result["baseline_metrics"],
            )

            st.subheader("テストデータでの予測結果")
            st.dataframe(result["compare_df"], use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(result["compare_df"]["日付"], result["compare_df"]["実績"], marker="o", label="実績")
            ax2.plot(result["compare_df"]["日付"], result["compare_df"]["予測"], marker="o", linestyle="--", label="予測")
            ax2.set_xlabel("日付")
            ax2.set_ylabel(target_col)
            ax2.set_title("実績と予測の比較")
            ax2.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

            st.subheader("特徴量の重要度")
            st.dataframe(result["importance_df"], use_container_width=True)
            render_importance_diagnosis(result["importance_df"])

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.bar(result["importance_df"]["特徴量"], result["importance_df"]["重要度"])
            ax3.set_ylabel("重要度")
            ax3.set_title("特徴量重要度")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)

            # 最終1点と比べると曜日変動に振り回されるため、直近1周期の平均を基準にする
            recent_cfg = GRANULARITY_CONFIG[detected_granularity]
            recent_window = recent_cfg["recent_window"]
            recent_label = recent_cfg["recent_label"]
            latest_actual = float(result["ts_df"][target_col].tail(recent_window).mean())

            forecast_summary = generate_forecast_summary(latest_actual, result["future_df"])
            render_forecast_summary(
                forecast_summary, target_col, result["importance_df"], recent_label
            )

            st.subheader("未来予測")
            future_df = result["future_df"].copy()
            future_df = future_df.rename(columns={date_col: "予測日付"})
            st.dataframe(future_df, use_container_width=True)

            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(result["ts_df"][date_col], result["ts_df"][target_col], marker="o", label="実績")
            ax4.plot(result["future_df"][date_col], result["future_df"]["予測値"], marker="o", linestyle="--", label="予測")
            ax4.set_xlabel("日付")
            ax4.set_ylabel(target_col)
            ax4.set_title("実績と未来予測")
            ax4.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig4)

            future_mean = float(result["future_df"]["予測値"].mean())
            st.subheader("補足コメント")
            st.success(generate_comment(latest_actual, future_mean, recent_label))

            export_df = result["future_df"].copy()
            export_df = export_df.rename(columns={date_col: "日付", "予測値": f"{target_col}_予測値"})
            csv_data = export_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "予測結果CSVをダウンロード",
                data=csv_data,
                file_name="forecast_result.csv",
                mime="text/csv",
            )

    except Exception as exc:
        show_user_error(f"予測処理で問題が発生しました。{exc}")

else:
    st.subheader("追加特徴量で予測")
    st.caption("日付以外の列を特徴量として使い、学習データの中で予測精度を確認します。")

    try:
        result = build_additional_feature_model(df, date_col, target_col)
    except Exception as exc:
        show_user_error(f"追加特徴量モデルの実行で問題が発生しました。{exc}")
        st.stop()

    render_accuracy_summary(
        result["mae"],
        result["rmse"],
        result["mape"],
        baseline_metrics=result["baseline_metrics"],
    )

    st.subheader("テストデータでの予測結果")
    st.dataframe(result["compare_df"], use_container_width=True)

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(result["compare_df"]["日付"], result["compare_df"]["実績"], marker="o", label="実績")
    ax5.plot(result["compare_df"]["日付"], result["compare_df"]["予測"], marker="o", linestyle="--", label="予測")
    ax5.set_xlabel("日付")
    ax5.set_ylabel(target_col)
    ax5.set_title("実績と予測の比較")
    ax5.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig5)

    st.subheader("特徴量の重要度")
    st.dataframe(result["importance_df"], use_container_width=True)
    render_importance_diagnosis(result["importance_df"])

    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.bar(result["importance_df"]["特徴量"], result["importance_df"]["重要度"])
    ax6.set_ylabel("重要度")
    ax6.set_title("特徴量重要度")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig6)
