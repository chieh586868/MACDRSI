import yfinance as yf
import pandas as pd
import os


class DataFetcher:
    def __init__(self, period: str = "6mo", interval: str = "1d"):
        self.period = period
        self.interval = interval

    def fetch(self, symbol: str) -> pd.DataFrame | None:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            if df.empty or len(df) < 60:
                return None
            df.index = pd.to_datetime(df.index)
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception:
            return None

    def fetch_batch(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            df = self.fetch(symbol)
            if df is not None:
                results[symbol] = df
        return results

    @staticmethod
    def load_symbols(filepath: str) -> list[str]:
        if not os.path.exists(filepath):
            return []
        symbols = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    symbols.append(line)
        return symbols
