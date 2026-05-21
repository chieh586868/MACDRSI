import pandas as pd
import numpy as np


class TechnicalIndicators:

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @classmethod
    def macd(
        cls,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        ema_fast = cls.ema(close, fast)
        ema_slow = cls.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = cls.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "hist": histogram,
        })

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def kd(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> pd.DataFrame:
        lowest = low.rolling(period).min()
        highest = high.rolling(period).max()
        rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        return pd.DataFrame({"K": k, "D": d})

    @classmethod
    def add_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        df["ma5"] = cls.sma(close, 5)
        df["ma20"] = cls.sma(close, 20)
        df["ma60"] = cls.sma(close, 60)

        macd_df = cls.macd(close)
        df["macd"] = macd_df["macd"]
        df["macd_signal"] = macd_df["signal"]
        df["macd_hist"] = macd_df["hist"]

        df["rsi"] = cls.rsi(close)

        kd_df = cls.kd(high, low, close)
        df["K"] = kd_df["K"]
        df["D"] = kd_df["D"]

        return df
