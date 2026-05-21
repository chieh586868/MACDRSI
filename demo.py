#!/usr/bin/env python3
"""
Demo 模式：使用模擬資料驗證選股邏輯
在沒有網路的環境下可執行此腳本確認程式正常運作。
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from screener.indicators import TechnicalIndicators
from screener.screener import StockScreener, ScreenRule


def generate_trending_stock(days: int = 120, trend: float = 0.001, seed: int = 0) -> pd.DataFrame:
    """產生帶有趨勢的模擬 K 線資料"""
    np.random.seed(seed)
    dates = [datetime.today() - timedelta(days=days - i) for i in range(days)]
    returns = np.random.normal(trend, 0.015, days)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
    volume = np.random.randint(1_000_000, 5_000_000, days)

    return pd.DataFrame({
        "Open": close * 0.998,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=pd.DatetimeIndex(dates))


def main():
    print("\n" + "=" * 60)
    print("  MACDRSI Demo 模式  |  使用模擬資料測試選股邏輯")
    print("=" * 60 + "\n")

    # 產生不同特性的模擬股票
    mock_stocks = {
        "上漲趨勢股A": generate_trending_stock(trend=0.003, seed=1),
        "上漲趨勢股B": generate_trending_stock(trend=0.002, seed=2),
        "橫盤震盪股C": generate_trending_stock(trend=0.0001, seed=3),
        "下跌趨勢股D": generate_trending_stock(trend=-0.002, seed=4),
        "急漲強勢股E": generate_trending_stock(trend=0.005, seed=5),
        "溫和上漲股F": generate_trending_stock(trend=0.001, seed=6),
    }

    # 展示各股票指標值
    print("各股票最新技術指標：")
    print("-" * 70)
    print(f"{'股票':<14} {'收盤價':>8} {'RSI':>6} {'MACD':>8} {'Signal':>8} {'K值':>6} {'D值':>6}")
    print("-" * 70)
    for name, df in mock_stocks.items():
        df = TechnicalIndicators.add_all(df.copy()).dropna()
        last = df.iloc[-1]
        print(
            f"{name:<14} {last['Close']:>8.2f} {last['rsi']:>6.1f} "
            f"{last['macd']:>8.4f} {last['macd_signal']:>8.4f} "
            f"{last['K']:>6.1f} {last['D']:>6.1f}"
        )

    print("\n")

    # 執行三種策略篩選
    for strategy_name, rule in [
        ("均衡策略（default）", ScreenRule(name="均衡策略", rsi_min=30, rsi_max=70, macd_golden_cross=True, ma_bullish=True, price_above_ma20=True)),
        ("突破策略（breakout）", ScreenRule(name="突破策略", rsi_min=50, rsi_max=80, macd_golden_cross=True, ma_bullish=True, price_above_ma20=True, volume_surge=True)),
        ("超賣反彈（oversold）", ScreenRule(name="超賣反彈", rsi_min=20, rsi_max=40, macd_golden_cross=False, kd_golden_cross=True, ma_bullish=False, price_above_ma20=False)),
    ]:
        screener = StockScreener(rule=rule)
        results = screener.screen(mock_stocks)
        print(f"策略：{strategy_name}  →  篩選結果：{len(results)} 支")
        for r in results:
            print(f"  ✓ {r.symbol:<14} 條件：{', '.join(r.matched_rules)}")
        if not results:
            print("  （無符合條件）")
        print()

    print("Demo 完成！實際使用請執行：python main.py --market tw")
    print("=" * 60)


if __name__ == "__main__":
    main()
