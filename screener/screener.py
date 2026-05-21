import pandas as pd
from dataclasses import dataclass, field
from .indicators import TechnicalIndicators


@dataclass
class ScreenRule:
    # RSI 範圍
    rsi_min: float = 30.0
    rsi_max: float = 70.0
    # MACD：macd 線 > signal 線（金叉）
    macd_golden_cross: bool = True
    # MACD histogram 由負轉正
    macd_hist_turning: bool = False
    # 均線多頭排列：ma5 > ma20 > ma60
    ma_bullish: bool = True
    # KD 金叉：K > D
    kd_golden_cross: bool = False
    # 收盤價在 ma20 之上
    price_above_ma20: bool = True
    # 成交量放大（當日量 > 5 日均量）
    volume_surge: bool = False
    # 自訂標籤（顯示用）
    name: str = "預設策略"


@dataclass
class ScreenResult:
    symbol: str
    close: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    K: float
    D: float
    ma5: float
    ma20: float
    ma60: float
    matched_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "代碼": self.symbol,
            "收盤價": round(self.close, 2),
            "RSI": round(self.rsi, 1),
            "MACD": round(self.macd, 4),
            "Signal": round(self.macd_signal, 4),
            "Hist": round(self.macd_hist, 4),
            "K值": round(self.K, 1),
            "D值": round(self.D, 1),
            "MA5": round(self.ma5, 2),
            "MA20": round(self.ma20, 2),
            "MA60": round(self.ma60, 2),
            "符合條件": ", ".join(self.matched_rules),
        }


class StockScreener:
    def __init__(self, rule: ScreenRule | None = None):
        self.rule = rule or ScreenRule()

    def _check(self, df: pd.DataFrame, symbol: str) -> ScreenResult | None:
        df = TechnicalIndicators.add_all(df)
        df = df.dropna()
        if len(df) < 2:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        matched = []

        rsi = last["rsi"]
        if self.rule.rsi_min <= rsi <= self.rule.rsi_max:
            matched.append(f"RSI({rsi:.1f})")

        macd = last["macd"]
        macd_sig = last["macd_signal"]
        if self.rule.macd_golden_cross:
            if macd > macd_sig and prev["macd"] <= prev["macd_signal"]:
                matched.append("MACD金叉")
            elif macd > macd_sig:
                matched.append("MACD多頭")

        if self.rule.macd_hist_turning:
            if last["macd_hist"] > 0 > prev["macd_hist"]:
                matched.append("Hist翻正")

        if self.rule.ma_bullish:
            if last["ma5"] > last["ma20"] > last["ma60"]:
                matched.append("均線多頭")

        if self.rule.kd_golden_cross:
            if last["K"] > last["D"] and prev["K"] <= prev["D"]:
                matched.append("KD金叉")

        if self.rule.price_above_ma20:
            if last["Close"] > last["ma20"]:
                matched.append("價>MA20")

        if self.rule.volume_surge:
            vol_ma5 = df["Volume"].tail(6).iloc[:-1].mean()
            if last["Volume"] > vol_ma5 * 1.5:
                matched.append("量增")

        if not matched:
            return None

        return ScreenResult(
            symbol=symbol,
            close=last["Close"],
            rsi=rsi,
            macd=macd,
            macd_signal=macd_sig,
            macd_hist=last["macd_hist"],
            K=last["K"],
            D=last["D"],
            ma5=last["ma5"],
            ma20=last["ma20"],
            ma60=last["ma60"],
            matched_rules=matched,
        )

    def screen(self, data: dict[str, pd.DataFrame]) -> list[ScreenResult]:
        results = []
        for symbol, df in data.items():
            result = self._check(df.copy(), symbol)
            if result:
                results.append(result)
        return results
