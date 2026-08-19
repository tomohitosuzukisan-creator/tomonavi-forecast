"""需要予測アプリの画面。

計算は core.py にまとめてあり、ここは「何を、どの順番で見せるか」に専念する。
"""
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analytics import track_event
from core import (
    ACTIVE_JP_FONT,
    CSV_ENCODINGS,
    GRANULARITY_CONFIG,
    GRANULARITY_LABELS,
    MAX_COLS,
    MAX_FEATURE_COLS,
    MAX_FILE_SIZE,
    MAX_FORECAST_PERIODS,
    MAX_ROWS,
    add_time_features,
    aggregate_timeseries,
    build_additional_feature_model,
    build_time_series_model,
    calculate_baseline_metrics,
    calculate_metrics,
    diagnose_feature_columns,
    diagnose_time_series_data,
    explain_feature_importance,
    explain_generic_importance,
    generate_comment,
    generate_forecast_summary,
    infer_granularity,
    is_ai_lost,
    normalize_target_series,
)


st.set_page_config(page_title="需要予測アプリ", layout="wide")


# アプリ内から1クリックで試せるサンプルデータ(いずれも合成データ)
_SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_FILES = {
    "ts": {
        "path": _SAMPLE_DIR / "sample_orders_daily.csv",
        "loaded": (
            "サンプルデータ（ある会社の日次受注件数・546日分の合成データ）を読み込みました。"
            "そのまま下の「予測する」ボタンまで進めます。自社のCSVをアップロードすると置き換わります。"
        ),
    },
    "bento": {
        "path": _SAMPLE_DIR / "sample_bento_daily.csv",
        "loaded": (
            "サンプルデータ（お弁当屋さんの平日販売データ・約9か月分の合成データ）を読み込みました。"
            "メニュー・天気・曜日などの手がかりから、何が販売数を動かしているかを見られます。"
            "自社のCSVをアップロードすると置き換わります。"
        ),
    },
}


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "use_sample" not in st.session_state:
    st.session_state["use_sample"] = False
if "sample_kind" not in st.session_state:
    st.session_state["sample_kind"] = "ts"

if not st.session_state.get("_ga_view_sent"):
    track_event("app_view")
    st.session_state["_ga_view_sent"] = True


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
    c1.metric("平均して何%ズレるか（MAPE）", "判定不可" if mape is None else f"±{mape:.1f}%")
    c2.metric("平均誤差（MAE）", f"{mae:.2f}")
    c3.metric("大きな外しの重み（RMSE）", f"{rmse:.2f}")

    st.caption(
        "まずは左のMAPE（±何%）を見てください。MAEは実際の数量単位での平均誤差、"
        "RMSEは大きく外した回をより重く数えた指標です。"
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
                f"過去データ上では、平均して±{mape:.1f}%程度のズレでした。データの期間・記録のされ方・外れ値などを見直すことをおすすめします。"
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
                st.error(f"AIモデルは、単純平均予測より誤差が大きくなっています。このデータでは単純な方法のほうが有利です。")

        if naive_improvement is not None:
            if naive_improvement > 5:
                st.success(f"AIモデルは、直前値をそのまま使う予測より約{naive_improvement:.1f}%誤差を改善しています。")
            elif naive_improvement >= -5:
                st.warning("AIモデルは、直前値をそのまま使う予測とほぼ同程度です。単純な方法でも十分な可能性があります。")
            else:
                st.error(f"AIモデルは、直前値をそのまま使う予測より誤差が大きくなっています。このデータでは単純な方法のほうが有利です。")

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


def render_consultation_cta(ai_lost: bool = False) -> None:
    """予測結果の後に、30分無料相談への導線を表示する。

    ai_lost: AIが単純な方法に負けた場合はTrue。悪い結果を「原因を一緒に見る理由」に変換する。
    """
    st.divider()
    if ai_lost:
        st.subheader("「予測しにくいデータ」には、必ず理由があります")
        st.markdown(
            "今回は、単純な方法のほうが当たる結果でした。実は、ここからが本題です。\n\n"
            "欠測や休業日の混ざり方、イレギュラーの多さ、記録のされ方——予測しにくさの原因は"
            "データごとに違います。その原因を突き止める作業こそ、30分無料相談でいちばんよく扱うテーマです。"
            "中小製造業のAI活用・需要予測を専門にしている中小企業診断士が、オンラインでお話を伺います。"
        )
    else:
        st.subheader("この結果、どう使えばいいか迷ったら")
        st.markdown(
            "予測は出せても「業務のどこに置くか」「この誤差は許容できるか」の判断は、"
            "データと会社の実情を知る人にしか下せません。\n\n"
            "中小製造業のAI活用・需要予測を専門にしている中小企業診断士が、"
            "30分無料でオンライン相談を承っています。"
        )
    st.link_button("30分無料相談を申し込む", "https://tomonavi.com/#contact")


def render_verdict(
    baseline_metrics: dict | None,
    future_df: pd.DataFrame | None,
    latest_actual: float,
    recent_label: str,
) -> None:
    """結論ファースト: このデータをAIで予測する価値があるかを、最初に一言で示す。"""
    st.subheader("結論")

    outlook = ""
    if (
        future_df is not None
        and not future_df.empty
        and "予測値" in future_df.columns
        and latest_actual
    ):
        future_mean = float(future_df["予測値"].mean())
        ratio = (future_mean - latest_actual) / latest_actual * 100
        outlook = (
            f"この先の平均は {future_mean:,.1f}"
            f"（{recent_label} {latest_actual:,.1f} に対して {ratio:+.1f}%）の見通しです。"
        )

    bm = baseline_metrics or {}
    model_mape = bm.get("model_mape")
    mean_mape = bm.get("mean_mape")
    naive_mape = bm.get("naive_mape")

    if model_mape is None or mean_mape is None or naive_mape is None:
        st.info("単純な方法との精度比較ができないデータのため、AIの優劣は判定できませんでした。" + outlook)
        return

    best_simple = min(mean_mape, naive_mape)
    best_simple_name = "単純平均" if mean_mape <= naive_mape else "直前値"

    if model_mape < best_simple * 0.95:
        st.success(
            f"**このデータは、AIで予測する価値があります。** "
            f"AIの誤差は平均 ±{model_mape:.1f}% で、単純な方法"
            f"（単純平均 ±{mean_mape:.1f}%・直前値 ±{naive_mape:.1f}%）より小さくなりました。"
            + outlook
        )
    elif model_mape <= best_simple * 1.05:
        st.info(
            f"**AIと単純な方法は、ほぼ互角でした。** "
            f"AIの誤差 ±{model_mape:.1f}% に対し、{best_simple_name}でも ±{best_simple:.1f}%。"
            "このデータなら、まず単純な方法から始めるのも十分な選択です。"
            + outlook
        )
    else:
        st.warning(
            f"**このデータでは、単純な方法のほうが当たっています。** "
            f"AIの誤差 ±{model_mape:.1f}% に対し、{best_simple_name}をそのまま使うほうが"
            f" ±{best_simple:.1f}% と小さい結果でした。無理にAIを使わない、も正しい判断です。"
            + outlook
        )


def render_importance_diagnosis(importance_df: pd.DataFrame) -> None:
    """特徴量重要度が出ない場合に理由を補足する。"""
    if importance_df.empty or "重要度" not in importance_df.columns:
        return

    total_importance = importance_df["重要度"].sum()
    if total_importance == 0:
        st.info(
            "今回のデータでは、どの手がかりが効いたかの内訳までは読み取れませんでした（予測そのものはできています）。"
            "データ件数が少ない場合や、数値の変動が小さい場合に起こります。"
            "期間を長くしたCSVで試すと、読み取れるようになることがあります。"
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


st.title("需要予測アプリ")
st.write("CSVファイルをアップロードして、日次・週次・月次の需要予測を行います。")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("最初からやり直す"):
        st.session_state["uploader_key"] += 1
        st.session_state["use_sample"] = False
        st.session_state["sample_kind"] = "ts"
        st.rerun()

st.info(
    "CSVの日付列と数量列を選ぶだけで予測できます。粒度(日次/週次/月次)は自動判定します。"
    "アップロードされたデータは保存しません。その場の計算にだけ使い、画面を閉じれば消えます。"
)

# 個人情報を含むファイルは扱わない旨を明示する。
# 予測に必要なのは日付と数量であり、氏名や連絡先は元々不要。
st.caption(
    "⚠️ 氏名・連絡先などの個人情報を含むファイルはアップロードしないでください。"
    "需要予測に必要なのは日付と数量、および曜日・天気などの条件だけです。"
)

uploaded_file = st.file_uploader(
    "CSVファイルをアップロードしてください",
    type=["csv"],
    key=f"file_uploader_{st.session_state['uploader_key']}",
)


if uploaded_file is None and not st.session_state["use_sample"]:
    st.info("まずはCSVファイルをアップロードしてください。手元にデータがなくても、下のボタンからサンプルで試せます。")
    sample_b1, sample_b2 = st.columns(2)
    with sample_b1:
        if st.button("受注データのサンプルで試す（日次・1年半分）", type="primary", use_container_width=True):
            track_event("app_sample_click", {"sample_kind": "ts"})
            st.session_state["use_sample"] = True
            st.session_state["sample_kind"] = "ts"
            st.rerun()
    with sample_b2:
        if st.button("お弁当屋さんのサンプルで試す（要因分析向け）", use_container_width=True):
            track_event("app_sample_click", {"sample_kind": "bento"})
            st.session_state["use_sample"] = True
            st.session_state["sample_kind"] = "bento"
            st.rerun()
    st.stop()


if uploaded_file is not None:
    # 実データがアップロードされたら、サンプル表示より優先する
    st.session_state["use_sample"] = False
    file_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("_ga_last_file") != file_key:
        track_event("app_file_upload", {"file_size_kb": round(uploaded_file.size / 1024, 1)})
        st.session_state["_ga_last_file"] = file_key
    try:
        validate_uploaded_file(uploaded_file)
        df = read_csv_with_fallbacks(uploaded_file)
    except Exception as exc:
        show_user_error(str(exc))
        st.stop()
else:
    sample_cfg = SAMPLE_FILES.get(st.session_state.get("sample_kind") or "ts", SAMPLE_FILES["ts"])
    try:
        df = pd.read_csv(sample_cfg["path"], encoding="utf-8-sig")
    except Exception:
        show_user_error("サンプルデータの読み込みに失敗しました。CSVをアップロードしてお試しください。")
        st.stop()
    st.success(sample_cfg["loaded"])


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
    show_user_error("少なくとも日付列と、予測したい数量の列の 2 列が必要です。")
    st.stop()

# データを見てから使い方を決める。2列(日付+数量)なら迷いようがないので時系列予測に直行する
if len(all_columns) == 2:
    mode = "時系列予測"
else:
    default_mode_idx = (
        1 if (st.session_state["use_sample"] and st.session_state.get("sample_kind") == "bento") else 0
    )
    mode_label = st.radio(
        "使い方を選んでください",
        [
            "日付と数量だけで予測する（時系列予測）",
            "他の列も使って、何が数字を動かしているかを見る（要因分析）",
        ],
        horizontal=True,
        index=default_mode_idx,
    )
    mode = "時系列予測" if (mode_label or "").startswith("日付と数量") else "追加特徴量で予測"

st.subheader("設定")
setting_col1, setting_col2, setting_col3 = st.columns(3)

with setting_col1:
    date_col = st.selectbox("日付列を選択", all_columns)

remaining_target_columns = [col for col in all_columns if col != date_col]
with setting_col2:
    target_col = st.selectbox("予測したい数量の列を選択", remaining_target_columns)

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
        c2.metric("合計", f"{df_time[target_col].sum():,.2f}")
        c3.metric("平均", f"{df_time[target_col].mean():,.2f}")

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
            track_event(
                "app_forecast_run",
                {"mode": "time_series", "granularity": detected_granularity, "rows": len(df_time)},
            )

            # 結論ファースト: 最初に「AIで予測する価値があるか」を一言で示す。
            # 比較基準は最終1点でなく直近1周期の平均(曜日・月末変動に振り回されないため)
            recent_cfg = GRANULARITY_CONFIG[detected_granularity]
            recent_window = recent_cfg["recent_window"]
            recent_label = recent_cfg["recent_label"]
            latest_actual = float(result["ts_df"][target_col].tail(recent_window).mean())
            render_verdict(
                result["baseline_metrics"], result["future_df"], latest_actual, recent_label
            )

            # 見通し(本編): サマリー → 未来予測グラフ → 補足コメント → ダウンロード
            forecast_summary = generate_forecast_summary(latest_actual, result["future_df"])
            render_forecast_summary(
                forecast_summary, target_col, result["importance_df"], recent_label
            )

            st.subheader("実績と未来予測")
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

            future_df = result["future_df"].copy()
            future_df = future_df.rename(columns={date_col: "予測日付"})
            st.dataframe(future_df, use_container_width=True)

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

            # 詳細は折りたたみに収納(結論・見通し・グラフだけで持ち帰れる構成にする)
            with st.expander("精度の詳しい数字を見る（単純な方法との比較・テスト期間の答え合わせ）"):
                render_accuracy_summary(
                    result["mae"],
                    result["rmse"],
                    result["mape"],
                    result["future_df"],
                    result["baseline_metrics"],
                )

                st.markdown("**テスト期間の実績と予測（答え合わせ）**")
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

            with st.expander("どの手がかりが効いているか（詳しい数字）"):
                if float(result["importance_df"]["重要度"].sum()) == 0:
                    render_importance_diagnosis(result["importance_df"])
                else:
                    display_importance = result["importance_df"].rename(columns={"特徴量": "手がかり"})
                    st.dataframe(display_importance, use_container_width=True)

                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.bar(display_importance["手がかり"], display_importance["重要度"])
                    ax3.set_ylabel("重要度")
                    ax3.set_title("手がかりの効き具合")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig3)

            render_consultation_cta(ai_lost=is_ai_lost(result["baseline_metrics"]))

    except Exception as exc:
        show_user_error(f"予測処理で問題が発生しました。{exc}")

else:
    st.subheader("要因分析（他の列も使った予測）")
    st.caption(
        "日付以外の列を手がかりとして使い、過去データの中で予測の精度と、どの手がかりが効いているかを確認します。"
        "このモードは未来の予測ではなく、要因の効き具合を見るためのものです。"
    )

    st.markdown("**使う手がかりを選んでください**")
    st.caption(
        "通し番号・自由記述・数量のコピーとみられる列は、結果を歪めるため理由を添えて最初からチェックを外してあります。"
    )
    feature_diagnoses = diagnose_feature_columns(df, date_col, target_col)
    selected_features = []
    for diag in feature_diagnoses:
        checked = st.checkbox(diag["col"], value=diag["default_on"], key=f"feature_{diag['col']}")
        if diag["note"]:
            st.caption(f"⚠️ {diag['note']}")
        if checked:
            selected_features.append(diag["col"])

    if len(selected_features) == 0:
        st.info("手がかりを1つ以上選んでください。")
        st.stop()

    if len(selected_features) > MAX_FEATURE_COLS:
        st.warning(
            f"一度に見る手がかりは {MAX_FEATURE_COLS} 個までにしています。"
            "多すぎると「どれが効いているか」が読み取れなくなるためです。"
            "業務の実感で効いていそうな列に絞ってみてください。"
        )
        st.stop()

    try:
        result = build_additional_feature_model(df, date_col, target_col, selected_features)
        track_event(
            "app_forecast_run",
            {"mode": "factor_analysis", "n_features": len(selected_features), "rows": len(df)},
        )
    except Exception as exc:
        show_user_error(f"要因分析の実行で問題が発生しました。{exc}")
        st.stop()

    st.subheader("どの手がかりが効いているか")

    # 注意つきの列をあえて含めた場合は、結果側でも読み方の注意を出す。
    # 例: 通し番号は時間の並びを丸暗記する道具になるため、重要度1位に見えても業務的な意味はない
    flagged_selected = [
        diag["col"] for diag in feature_diagnoses if diag["note"] and diag["col"] in selected_features
    ]
    if flagged_selected:
        st.warning(
            f"⚠️ 注意つきの列（{'、'.join(flagged_selected)}）を手がかりに含めています。"
            "通し番号のような列は、行の並び（＝時間）を丸暗記する道具になるため重要度が高く出やすいですが、"
            "未来のデータでは役に立たず、業務で使える要因でもありません。"
            "外した場合の結果と見比べてみてください。"
        )
    if float(result["importance_df"]["重要度"].sum()) == 0:
        render_importance_diagnosis(result["importance_df"])
    else:
        display_importance = result["importance_df"].rename(columns={"特徴量": "手がかり"})
        st.dataframe(display_importance, use_container_width=True)

        fig6, ax6 = plt.subplots(figsize=(8, 4))
        ax6.bar(display_importance["手がかり"], display_importance["重要度"])
        ax6.set_ylabel("重要度")
        ax6.set_title("手がかりの効き具合")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig6)

        explain_lines = explain_generic_importance(result["importance_df"], target_col)
        if explain_lines:
            st.markdown("**読み取れること（参考）**")
            for line in explain_lines:
                st.markdown(f"・{line}")
            st.caption("※統計的な傾向にもとづく参考情報であり、因果関係を示すものではありません。")

    with st.expander("精度の詳しい数字を見る（単純な方法との比較・テスト期間の答え合わせ）"):
        render_accuracy_summary(
            result["mae"],
            result["rmse"],
            result["mape"],
            baseline_metrics=result["baseline_metrics"],
        )

        st.markdown("**テスト期間の実績と予測（答え合わせ）**")
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

    render_consultation_cta(ai_lost=is_ai_lost(result["baseline_metrics"]))
