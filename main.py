#!/usr/bin/env python3
"""
MACDRSI 選股程式
支援台股（.TW）與美股，使用 MACD、RSI、KD、均線等技術指標篩選。
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from screener import DataFetcher, StockScreener
from screener.screener import ScreenRule


def parse_args():
    parser = argparse.ArgumentParser(
        description="MACDRSI 技術分析選股工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
策略模式:
  default   RSI 30~70 + MACD 多頭 + 均線多頭 + 價>MA20
  breakout  RSI 50~80 + MACD 金叉 + 均線多頭 + 量增
  oversold  RSI 20~40 + KD 金叉（超賣反彈）

範例:
  python main.py --market tw
  python main.py --market us --strategy breakout
  python main.py --symbols AAPL MSFT NVDA
  python main.py --market both --output result.csv
        """,
    )
    parser.add_argument(
        "--market", choices=["tw", "us", "both"], default="both",
        help="選擇市場（tw/us/both）",
    )
    parser.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL",
        help="直接指定股票代碼，例：--symbols 2330.TW AAPL",
    )
    parser.add_argument(
        "--strategy", choices=["default", "breakout", "oversold"], default="default",
        help="選股策略",
    )
    parser.add_argument(
        "--period", default="6mo",
        help="資料期間（1mo/3mo/6mo/1y），預設 6mo",
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="輸出 CSV 檔案路徑",
    )
    parser.add_argument(
        "--rsi-min", type=float, default=None, help="RSI 下限（覆蓋策略預設）",
    )
    parser.add_argument(
        "--rsi-max", type=float, default=None, help="RSI 上限（覆蓋策略預設）",
    )
    return parser.parse_args()


STRATEGIES: dict[str, ScreenRule] = {
    "default": ScreenRule(
        name="均衡策略",
        rsi_min=30, rsi_max=70,
        macd_golden_cross=True,
        ma_bullish=True,
        price_above_ma20=True,
    ),
    "breakout": ScreenRule(
        name="突破策略",
        rsi_min=50, rsi_max=80,
        macd_golden_cross=True,
        macd_hist_turning=False,
        ma_bullish=True,
        price_above_ma20=True,
        volume_surge=True,
    ),
    "oversold": ScreenRule(
        name="超賣反彈",
        rsi_min=20, rsi_max=40,
        macd_golden_cross=False,
        kd_golden_cross=True,
        ma_bullish=False,
        price_above_ma20=False,
    ),
}


def collect_symbols(args) -> list[str]:
    if args.symbols:
        return args.symbols

    symbols = []
    if args.market in ("tw", "both"):
        tw_file = os.path.join(BASE_DIR, "stocks", "tw_stocks.txt")
        symbols += DataFetcher.load_symbols(tw_file)
    if args.market in ("us", "both"):
        us_file = os.path.join(BASE_DIR, "stocks", "us_stocks.txt")
        symbols += DataFetcher.load_symbols(us_file)
    return symbols


def main():
    args = parse_args()
    rule = STRATEGIES[args.strategy]

    if args.rsi_min is not None:
        rule.rsi_min = args.rsi_min
    if args.rsi_max is not None:
        rule.rsi_max = args.rsi_max

    symbols = collect_symbols(args)
    if not symbols:
        print("錯誤：找不到任何股票代碼。")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  MACDRSI 選股程式  |  策略：{rule.name}")
    print(f"  市場：{args.market.upper()}  |  股票數：{len(symbols)}")
    print(f"  RSI 範圍：{rule.rsi_min} ~ {rule.rsi_max}")
    print(f"  資料期間：{args.period}")
    print(f"{'='*60}\n")

    fetcher = DataFetcher(period=args.period)
    screener = StockScreener(rule=rule)

    print(f"正在下載 {len(symbols)} 支股票資料...")
    data = {}
    for i, sym in enumerate(symbols, 1):
        df = fetcher.fetch(sym)
        if df is not None:
            data[sym] = df
        status = "✓" if df is not None else "✗"
        print(f"  [{i:3d}/{len(symbols)}] {sym:<12} {status}", end="\r")
    print(f"\n下載完成：成功 {len(data)} / {len(symbols)} 支\n")

    results = screener.screen(data)

    if not results:
        print("沒有符合條件的股票。")
        return

    rows = [r.to_dict() for r in results]
    df_out = pd.DataFrame(rows)

    try:
        from tabulate import tabulate
        print(tabulate(df_out, headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        print(df_out.to_string(index=False))

    print(f"\n共找到 {len(results)} 支符合條件的股票。")
    print(f"掃描時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(BASE_DIR, "output", args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"結果已儲存至：{out_path}")


if __name__ == "__main__":
    main()
