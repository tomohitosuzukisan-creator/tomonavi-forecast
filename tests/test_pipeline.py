"""時系列予測の一連の流れ(粒度判定→集計→診断→学習→予測→指標)を検証する。"""
import pandas as pd
import pytest


def _run(app, df, date_col, target_col, periods=7):
    granularity = app.infer_granularity(df[date_col])
    return granularity, app.build_time_series_model(
        df=df,
        date_col=date_col,
        target_col=target_col,
        granularity=granularity,
        forecast_periods=periods,
    )


class TestGranularity:
    def test_日次データを日次と判定する(self, app, orders_df):
        assert app.infer_granularity(orders_df["日付"]) == "daily"

    def test_週次データを週次と判定する(self, app, weekly_df):
        assert app.infer_granularity(weekly_df["date"]) == "weekly"

    def test_月次データを月次と判定する(self, app, monthly_df):
        assert app.infer_granularity(monthly_df["date"]) == "monthly"


class TestTimeSeriesModel:
    @pytest.fixture(scope="class")
    @staticmethod
    def result(app, orders_df):
        return _run(app, orders_df, "日付", "受注件数")[1]

    def test_指定した期間数だけ未来を予測する(self, result):
        assert len(result["future_df"]) == 7

    def test_精度指標が算出される(self, result):
        assert result["mae"] is not None
        assert result["rmse"] is not None
        assert result["mape"] is not None

    def test_予測結果をCSVに書き出せる(self, result):
        csv = result["future_df"].to_csv(index=False, encoding="utf-8-sig")
        assert len(csv) > 0

    def test_未来日付が実績の最終日より後になる(self, result):
        last_actual = result["ts_df"]["日付"].max()
        assert result["future_df"]["日付"].min() > last_actual


class TestBaselineComparison:
    """このアプリの要。AIと単純な方法を必ず比較し、負けたら負けたと出せること。"""

    def test_構造のあるデータではAIが単純な方法に勝つ(self, app, orders_df):
        bm = _run(app, orders_df, "日付", "受注件数")[1]["baseline_metrics"]
        assert bm["model_mape"] < bm["mean_mape"]
        assert bm["model_mape"] < bm["naive_mape"]

    def test_ベースライン指標が3種類そろう(self, app, orders_df):
        bm = _run(app, orders_df, "日付", "受注件数")[1]["baseline_metrics"]
        assert {"model_mape", "mean_mape", "naive_mape"} <= set(bm)

    def test_AIが勝っていれば負け判定にならない(self, app, orders_df):
        bm = _run(app, orders_df, "日付", "受注件数")[1]["baseline_metrics"]
        assert app.is_ai_lost(bm) is False

    def test_AIが負けていれば負け判定になる(self, app):
        assert app.is_ai_lost({"model_mape": 30.0, "mean_mape": 20.0, "naive_mape": 25.0}) is True

    def test_比較できないデータでは負け判定にしない(self, app):
        assert app.is_ai_lost({"model_mape": None, "mean_mape": None, "naive_mape": None}) is False
        assert app.is_ai_lost(None) is False


class TestDataDiagnosis:
    def test_良質なデータには成功メッセージを返す(self, app, orders_df):
        ts = app.aggregate_timeseries(orders_df, "日付", "受注件数", "daily")
        levels = [m["level"] for m in app.diagnose_time_series_data(ts, "日付", "受注件数", "daily")]
        assert "success" in levels

    def test_件数不足のデータには警告を返す(self, app, small_df):
        ts = app.aggregate_timeseries(small_df, "日付", "売上", "daily")
        levels = [m["level"] for m in app.diagnose_time_series_data(ts, "日付", "売上", "daily")]
        assert "warning" in levels or "error" in levels

    def test_件数不足では学習を中断する(self, app, small_df):
        with pytest.raises(ValueError, match="行以上"):
            app.build_time_series_model(
                df=small_df, date_col="日付", target_col="売上",
                granularity="daily", forecast_periods=7,
            )


class TestSmallData:
    """小規模データでも解釈できること(min_child_samples調整の回帰テスト)。"""

    def test_月次データでも手がかりの効き具合が出る(self, app, monthly_df):
        result = _run(app, monthly_df, "date", "sales")[1]
        assert result["importance_df"]["重要度"].sum() > 0, (
            "月次のような小規模データで重要度が全ゼロになると、画面が壊れて見える"
        )

    def test_週次データでも予測できる(self, app, weekly_df):
        result = _run(app, weekly_df, "date", "sales")[1]
        assert len(result["future_df"]) == 7
        assert result["mape"] is not None
