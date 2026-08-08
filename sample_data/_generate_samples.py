"""サンプルCSVを生成するスクリプト(開発用・再現可能)。

2種類を用意する。それぞれ「診断結果が違う」ことに意味がある。

1. sample_orders_daily.csv   … 構造(曜日・月・トレンド)がはっきりしたデータ。AIが単純な方法に勝つ
2. sample_small_25rows.csv   … 件数が少ないデータ。品質診断が警告を出す

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
