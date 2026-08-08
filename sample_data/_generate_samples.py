"""サンプルCSVを生成するスクリプト(開発用・再現可能)。

3種類を用意する。それぞれ「診断結果が違う」ことに意味がある。

1. sample_orders_daily.csv   … 構造(曜日・月・トレンド)がはっきりしたデータ。AIが単純な方法に勝つ(時系列予測用)
2. sample_small_25rows.csv   … 件数が少ないデータ。品質診断が警告を出す
3. sample_bento_daily.csv    … お弁当屋さんの日次販売データ(合成)。メニュー・天気・曜日などの
                                手がかり列を持ち、追加特徴量(要因分析)モードのサンプルになる。
                                ※SIGNATEのお弁当コンペを着想元にした自作データ。実データは規約上再配布不可

実行: .venv/Scripts/python.exe sample_data/_generate_samples.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(20260808)


# ------------------------------------------------------------------
# 1. 中小製造業の日次受注件数(構造がはっきりしたデータ)
# ------------------------------------------------------------------
dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")

# 曜日効果: 平日が高く、土日は少ない(BtoB製造業の受注らしい形)
dow_effect = {0: 1.05, 1: 1.00, 2: 0.98, 3: 1.00, 4: 1.12, 5: 0.35, 6: 0.28}

# 月効果: 3月(期末)と12月が高く、8月(夏季休暇)と1月が低い
month_effect = {
    1: 0.82, 2: 0.95, 3: 1.25, 4: 0.92, 5: 0.96, 6: 1.02,
    7: 1.05, 8: 0.72, 9: 1.08, 10: 1.04, 11: 1.06, 12: 1.15,
}

base = 100.0
values = []
for i, d in enumerate(dates):
    trend = 1.0 + 0.00035 * i                      # 緩やかな増加
    monthend = 1.18 if d.day >= 26 else 1.0        # 月末の駆け込み
    v = base * trend * dow_effect[d.dayofweek] * month_effect[d.month] * monthend
    v += rng.normal(0, 5.0)                        # 独立ノイズ(構造に対して小さめ)
    values.append(max(int(round(v)), 0))

orders = pd.DataFrame({"日付": dates.strftime("%Y-%m-%d"), "受注件数": values})
orders.to_csv(OUT / "sample_orders_daily.csv", index=False, encoding="utf-8-sig")
print(f"sample_orders_daily.csv    : {len(orders)}行")


# ------------------------------------------------------------------
# 2. 件数が少ないデータ(品質診断の警告を確認する用)
# ------------------------------------------------------------------
small_dates = pd.date_range("2026-01-01", periods=25, freq="D")
small = [120 + int(rng.normal(0, 12)) for _ in range(25)]
pd.DataFrame({"日付": small_dates.strftime("%Y-%m-%d"), "売上": small}).to_csv(
    OUT / "sample_small_25rows.csv", index=False, encoding="utf-8-sig"
)
print(f"sample_small_25rows.csv    : 25行")


# ------------------------------------------------------------------
# 3. お弁当屋さんの日次販売データ(要因分析・追加特徴量モード用)
#    オフィス街の弁当屋の平日ランチ販売を想定。効き方は設計値:
#    メニュー > 天気 > 曜日 の順で効く。気温は真夏に少し下がる程度。
# ------------------------------------------------------------------
bento_days = [d for d in pd.date_range("2025-10-01", "2026-06-30", freq="D") if d.dayofweek < 5]

menus = ["からあげ弁当", "ハンバーグ弁当", "焼き魚弁当", "カレー弁当", "生姜焼き弁当"]
menu_effect = {"からあげ弁当": 1.25, "ハンバーグ弁当": 1.05, "焼き魚弁当": 0.80,
               "カレー弁当": 1.10, "生姜焼き弁当": 0.95}
menu_price = {"からあげ弁当": 590, "ハンバーグ弁当": 620, "焼き魚弁当": 560,
              "カレー弁当": 540, "生姜焼き弁当": 600}
weather_effect = {"晴れ": 1.00, "くもり": 0.95, "雨": 0.75}
bento_dow_effect = {0: 0.95, 1: 1.00, 2: 1.00, 3: 1.05, 4: 1.15}
dow_names = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}
# 月別の平均気温(さいたま近辺のざっくりした値)
month_temp = {1: 5, 2: 6, 3: 10, 4: 15, 5: 20, 6: 23, 7: 27, 8: 29, 9: 25, 10: 18, 11: 12, 12: 7}

bento_rows = []
bento_base = 90.0
for d in bento_days:
    menu = menus[int(rng.integers(0, len(menus)))]
    weather = ["晴れ", "くもり", "雨"][int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
    temp = round(month_temp[d.month] + float(rng.normal(0, 3.0)), 1)
    event = "あり" if rng.random() < 0.05 else "なし"

    v = bento_base
    v *= menu_effect[menu]
    v *= weather_effect[weather]
    v *= bento_dow_effect[d.dayofweek]
    if temp >= 30:                      # 真夏日は食欲が落ちて少し減る
        v *= 0.90
    if event == "あり":                 # 近隣イベントの日は増える
        v *= 1.30
    v += float(rng.normal(0, 6.0))

    bento_rows.append({
        "日付": d.strftime("%Y-%m-%d"),
        "販売数": max(int(round(v)), 0),
        "メニュー": menu,
        "曜日": dow_names[d.dayofweek],
        "天気": weather,
        "気温": temp,
        "イベント": event,
        "価格": menu_price[menu] + int(rng.integers(-1, 2)) * 10,  # ほぼメニュー連動+日々の端数
    })

bento = pd.DataFrame(bento_rows)
bento.to_csv(OUT / "sample_bento_daily.csv", index=False, encoding="utf-8-sig")
print(f"sample_bento_daily.csv     : {len(bento)}行")
