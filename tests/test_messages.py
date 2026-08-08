"""利用者に見せる文章の検証。

このアプリは「伝わること」と「正直であること」が主な役割なので、
数字そのものと同じくらい、文言の分岐が壊れていないかが重要になる。
"""
import pandas as pd
import pytest


class TestForecastSummary:
    def _summary(self, core, latest, values):
        future = pd.DataFrame({"予測値": values})
        return core.generate_forecast_summary(latest, future)

    def test_増加傾向を判定する(self, core):
        s = self._summary(core, 100.0, [130.0] * 7)
        assert s["change_ratio"] > 0
        assert "増" in s["trend"] or "増" in s["trend_message"]

    def test_減少傾向を判定する(self, core):
        s = self._summary(core, 100.0, [70.0] * 7)
        assert s["change_ratio"] < 0
        assert "減" in s["trend"] or "減" in s["trend_message"]

    def test_横ばいを判定する(self, core):
        s = self._summary(core, 100.0, [100.0] * 7)
        assert abs(s["change_ratio"]) < 0.05

    def test_空の予測でも落ちない(self, core):
        s = core.generate_forecast_summary(100.0, pd.DataFrame({"予測値": []}))
        assert s["trend"] == "判定不可"


class TestComment:
    def test_基準ラベルが文中に出る(self, core):
        text = core.generate_comment(100.0, 90.0, "直近4週の平均")
        assert "直近4週の平均" in text

    def test_実績が0でも落ちずに理由を説明する(self, core):
        text = core.generate_comment(0.0, 90.0, "直近4週の平均")
        assert "0" in text
        assert len(text) > 10

    def test_業務アクションを断定で指示しない(self, core):
        """予測値は参考情報。「発注を増やせ」のような業務指示はしない方針。

        なお「データにない情報も必ず併せて確認してください」のような
        注意喚起は、断定ではなく利用者を守る文なので許容する。
        """
        text = core.generate_comment(100.0, 130.0, "直近4週の平均")
        for ng in ["発注を増やし", "発注してください", "仕入れを増やし", "在庫を増やし", "確実に売れ"]:
            assert ng not in text

    def test_機械的な算出であることを断る(self, core):
        text = core.generate_comment(100.0, 130.0, "直近4週の平均")
        assert "機械的" in text or "パターン" in text


class TestRecentWindow:
    """比較基準を「最終1点」から「直近1周期の平均」に変えた修正の回帰テスト。

    月末の平日1日と、土日を含む予測期間を比べて -40% と誤警告した不具合の再発防止。
    """

    def test_全粒度に直近窓の設定がある(self, core):
        for key, cfg in core.GRANULARITY_CONFIG.items():
            assert cfg["recent_window"] > 1, f"{key} の比較基準が1点になっている"
            assert cfg["recent_label"]

    def test_日次は4週で均す(self, core):
        assert core.GRANULARITY_CONFIG["daily"]["recent_window"] == 28

    def test_日次サンプルで増減率が過大にならない(self, core, orders_df):
        result = core.build_time_series_model(
            df=orders_df, date_col="日付", target_col="受注件数",
            granularity="daily", forecast_periods=7,
        )
        cfg = core.GRANULARITY_CONFIG["daily"]
        latest = float(result["ts_df"]["受注件数"].tail(cfg["recent_window"]).mean())
        future_mean = float(result["future_df"]["予測値"].mean())
        ratio = (future_mean - latest) / latest
        assert abs(ratio) < 0.20, "曜日や月末の偏りで実態のない大幅増減が出ている"


class TestFeatureImportanceExplanation:
    def test_時系列の手がかりを平易な日本語にする(self, core, orders_df):
        result = core.build_time_series_model(
            df=orders_df, date_col="日付", target_col="受注件数",
            granularity="daily", forecast_periods=7,
        )
        lines = core.explain_feature_importance(result["importance_df"])
        assert lines
        # lag7 のような専門用語がそのまま出ていないこと
        assert not any("lag" in line for line in lines)
