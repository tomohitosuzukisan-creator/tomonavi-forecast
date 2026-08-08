"""要因分析モード(手がかり列を使った分析)の検証。

このモードは「予測できてしまう罠」を教えるのが役割なので、
ガードが正しく働くことを重点的に確認する。
"""
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def diagnoses(core, bento_df):
    return core.diagnose_feature_columns(bento_df, "日付", "販売数")


@pytest.fixture(scope="module")
def default_features(diagnoses):
    """アプリが既定でONにする手がかり(=ユーザーが何もしなければ使われる列)。"""
    return [d["col"] for d in diagnoses if d["default_on"]]


class TestColumnGuards:
    def test_通し番号はID疑いとして既定で外す(self, diagnoses):
        diag = next(d for d in diagnoses if d["col"] == "注文伝票No")
        assert diag["default_on"] is False
        assert "通し番号" in diag["note"] or "ID" in diag["note"]

    def test_外した列には理由が添えられる(self, diagnoses):
        for diag in diagnoses:
            if not diag["default_on"]:
                assert diag["note"], f"{diag['col']} を外した理由が空になっている"

    def test_通常の手がかりは既定で使う(self, default_features):
        assert {"メニュー", "曜日", "天気", "気温"} <= set(default_features)

    def test_目的変数のコピーはリーク疑いとして外す(self, core, bento_df):
        leaked = bento_df.copy()
        leaked["売上金額"] = leaked["販売数"] * 600  # 目的変数から作った列
        diag = next(
            d for d in core.diagnose_feature_columns(leaked, "日付", "販売数")
            if d["col"] == "売上金額"
        )
        assert diag["default_on"] is False
        assert "リーク" in diag["note"] or "コピー" in diag["note"]

    def test_種類が多すぎる文字列は自由記述疑いとして外す(self, core, bento_df):
        wordy = bento_df.copy()
        wordy["備考"] = [f"メモ{i}" for i in range(len(wordy))]
        diag = next(
            d for d in core.diagnose_feature_columns(wordy, "日付", "販売数")
            if d["col"] == "備考"
        )
        assert diag["default_on"] is False


class TestFactorModel:
    @pytest.fixture(scope="class")
    @staticmethod
    def result(core, bento_df, default_features):
        return core.build_additional_feature_model(bento_df, "日付", "販売数", default_features)

    def test_精度指標が算出される(self, result):
        assert result["mape"] is not None

    def test_手がかりの効き具合が出る(self, result):
        assert result["importance_df"]["重要度"].sum() > 0

    def test_カテゴリ列も効き具合に反映される(self, result):
        """cat.codes をやめてカテゴリ型にした効果。メニューや曜日が死ななくなった。"""
        imp = dict(zip(result["importance_df"]["特徴量"], result["importance_df"]["重要度"]))
        assert imp.get("メニュー", 0) > 0
        assert imp.get("曜日", 0) > 0

    def test_AIが単純平均に勝つ(self, result):
        bm = result["baseline_metrics"]
        assert bm["model_mape"] < bm["mean_mape"]

    def test_行の順番が違っても結果は変わらない(self, core, bento_df, default_features, result):
        """日付ソートの回帰テスト。CSVの並び順で結果が変わってはいけない。"""
        shuffled = bento_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        other = core.build_additional_feature_model(shuffled, "日付", "販売数", default_features)
        assert other["mape"] == pytest.approx(result["mape"])

    def test_手がかりが1つもなければ中断する(self, core, bento_df):
        with pytest.raises(ValueError, match="手がかり"):
            core.build_additional_feature_model(bento_df, "日付", "販売数", [])

    def test_文字列の手がかりだけでも動く(self, core, bento_df):
        """pandas 3.x で文字列列が全行NaN化して落ちていた不具合の回帰テスト。"""
        result = core.build_additional_feature_model(
            bento_df, "日付", "販売数", ["メニュー", "天気", "曜日"]
        )
        assert result["mape"] is not None
        assert result["importance_df"]["重要度"].sum() > 0


class TestExplanation:
    def test_効き具合の解釈文が生成される(self, core, bento_df, default_features):
        result = core.build_additional_feature_model(bento_df, "日付", "販売数", default_features)
        lines = core.explain_generic_importance(result["importance_df"], "販売数")
        assert lines
        assert "販売数" in lines[0]

    def test_効き具合がなければ解釈文は空になる(self, core):
        empty = pd.DataFrame({"特徴量": ["a", "b"], "重要度": [0, 0]})
        assert core.explain_generic_importance(empty, "販売数") == []
