"""威廉波浪選股 v4（嚴格版）

設計理念（依使用者範例 2455/2355/8081/2313）：
  好的選股 = 「深底 → 8-21 天 5-7 波 → 今天大漲脫離超賣」

三段嚴格判定（必須同時成立）：
  1. 深底：過去 8-21 個交易日內，WR 曾觸及 ≤ -80
  2. 波浪：從深底到今天，WR 形成 ≥ 2 個局部高點（5-7 波結構）
  3. 脫離：今天 WR > -50 且 > 昨天 WR（確認向上脫離超賣）

舊版「simple_oversold」「bottom_bounce」「W底」保留但降為次要訊號，
不再單獨觸發買訊（priority < 4），主要訊號只在 WAVE_COMPLETE 觸發。
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────
# Williams %R 計算
# ──────────────────────────────────────────────────────────
def calc_williams_r(highs: pd.Series, lows: pd.Series,
                    closes: pd.Series, period: int = 20) -> pd.Series:
    h = highs.rolling(period).max()
    l = lows.rolling(period).min()
    return (h - closes) / (h - l + 1e-9) * -100


# ──────────────────────────────────────────────────────────
# 嚴格波浪偵測：深底 → 波浪 → 脫離
# ──────────────────────────────────────────────────────────
def detect_wave_complete(sl: np.ndarray,
                          deep_low_thr: float = -80.0,
                          window_min: int = 8,
                          window_max: int = 21,
                          min_peaks: int = 2,
                          escape_thr: float = -50.0) -> dict:
    """
    回傳 dict：
      ok            : True 表示三段條件全滿足
      deep_low_val  : 深底的 WR 值
      bars_from_low : 深底距今天幾根 K
      peak_count    : 深底到今天的局部高點數
      wr_now        : 今天 WR
      wr_yest       : 昨天 WR
    """
    n = len(sl)
    out = {"ok": False, "deep_low_val": 0.0, "bars_from_low": 0,
           "peak_count": 0, "wr_now": -50.0, "wr_yest": -50.0}
    if n < window_max + 2:
        return out

    wr_now  = float(sl[-1])
    wr_yest = float(sl[-2])
    out["wr_now"]  = round(wr_now, 2)
    out["wr_yest"] = round(wr_yest, 2)

    # 段3 先檢查（最便宜）：今天必須脫離超賣且向上
    if wr_now <= escape_thr:        return out
    if wr_now <= wr_yest:           return out

    # 段1：在 [n-1-window_max, n-1-window_min] 區間找最低 WR
    lo_idx_start = max(0, n - 1 - window_max)
    lo_idx_end   = n - 1 - window_min  # exclusive of window_min..now
    if lo_idx_end <= lo_idx_start:
        return out
    window = sl[lo_idx_start:lo_idx_end + 1]
    if np.all(np.isnan(window)):
        return out
    rel_min_idx = int(np.nanargmin(window))
    deep_low_idx = lo_idx_start + rel_min_idx
    deep_low_val = float(window[rel_min_idx])
    if deep_low_val > deep_low_thr:
        return out
    bars_from_low = n - 1 - deep_low_idx
    if not (window_min <= bars_from_low <= window_max):
        return out

    # 段2：深底到今天的區段內，數局部高點（WIN=1：嚴格鄰居比較）
    sub = sl[deep_low_idx:n]
    m = len(sub)
    if m < 4:
        return out
    peak_count = 0
    WIN = 1
    last_peak_pos = -10
    for i in range(WIN, m - WIN):
        v = sub[i]
        if np.isnan(v):
            continue
        left  = sub[i - WIN:i]
        right = sub[i + 1:i + WIN + 1]
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            continue
        if v >= np.max(left) and v >= np.max(right):
            # 避免緊鄰雙峰被計兩次
            if i - last_peak_pos >= 2:
                peak_count += 1
                last_peak_pos = i
    if peak_count < min_peaks:
        out["peak_count"] = peak_count
        out["deep_low_val"] = round(deep_low_val, 2)
        out["bars_from_low"] = bars_from_low
        return out

    out.update({
        "ok": True,
        "deep_low_val": round(deep_low_val, 2),
        "bars_from_low": bars_from_low,
        "peak_count": peak_count,
    })
    return out


# ──────────────────────────────────────────────────────────
# 舊版波浪結構偵測（保留作為次要資訊，不再驅動買訊）
# ──────────────────────────────────────────────────────────
def find_wr_waves_v4(wr: pd.Series, lookback: int = 60,
                     debug: bool = False) -> dict:
    empty = {
        "wave_count": 0, "current_wave": 0, "direction": "unknown",
        "wr_now": -50.0, "bars_since_last": 0, "last_peak_val": -50.0,
        "peaks": [], "troughs": [],
    }
    if len(wr) < 10:
        return empty
    wr_clean = wr.dropna()
    if len(wr_clean) < 10:
        return empty
    actual_lb = min(lookback, len(wr_clean))
    sl = wr_clean.iloc[-actual_lb:].values.astype(float)
    n  = len(sl)
    wr_now = float(sl[-1])

    WIN = 1 if n < 40 else 2
    peaks = []; troughs = []
    for i in range(WIN, n - WIN):
        if np.isnan(sl[i]): continue
        left  = sl[i - WIN:i]
        right = sl[i + 1:i + WIN + 1]
        if np.any(np.isnan(left)) or np.any(np.isnan(right)): continue
        v = sl[i]
        if v >= np.max(left) and v >= np.max(right):
            if not peaks or (i - peaks[-1]) >= 2:
                peaks.append(i)
        elif v <= np.min(left) and v <= np.min(right):
            if not troughs or (i - troughs[-1]) >= 2:
                troughs.append(i)

    turns = sorted([(i, "peak") for i in peaks] + [(i, "trough") for i in troughs])
    filtered = []
    for t in turns:
        if filtered and filtered[-1][1] == t[1]:
            if t[1] == "peak":
                if sl[t[0]] > sl[filtered[-1][0]]: filtered[-1] = t
            else:
                if sl[t[0]] < sl[filtered[-1][0]]: filtered[-1] = t
        else:
            if filtered and abs(sl[t[0]] - sl[filtered[-1][0]]) < 2:
                continue
            filtered.append(t)

    wave_count = len(filtered)
    if wave_count == 0:
        return {**empty, "wr_now": round(wr_now, 2), "peaks": peaks, "troughs": troughs}

    last_type = filtered[-1][1]
    last_idx  = filtered[-1][0]
    direction = "down" if last_type == "peak" else "up"
    bsl = n - 1 - last_idx
    last_peak_val = -50.0
    for f in reversed(filtered):
        if f[1] == "peak":
            last_peak_val = float(sl[f[0]]); break

    return {
        "wave_count": wave_count, "current_wave": wave_count + 1,
        "direction": direction, "wr_now": round(wr_now, 2),
        "bars_since_last": bsl, "last_peak_val": round(last_peak_val, 2),
        "peaks": peaks, "troughs": troughs,
    }


# ──────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────
def calc_williams_signals_v4(closes: pd.Series,
                              highs:  pd.Series,
                              lows:   pd.Series,
                              ind:    dict = None,
                              debug:  bool = False) -> dict:
    if len(closes) < 30:
        return {"buy_signals": [], "wave_info": {}, "wr_now": -50.0, "wr_list": []}

    if ind is not None and "wr" in ind:
        wr = ind["wr"]
    else:
        wr = calc_williams_r(highs, lows, closes, period=20)

    wr_clean = wr.dropna()
    if len(wr_clean) < 25:
        return {"buy_signals": [], "wave_info": {}, "wr_now": -50.0, "wr_list": []}

    sl = wr_clean.iloc[-60:].values.astype(float) if len(wr_clean) >= 60 \
         else wr_clean.values.astype(float)

    # 嚴格波浪完成偵測（主要訊號）
    wc = detect_wave_complete(sl)

    # 次要資訊（波浪結構供前端顯示用，不驅動買訊）
    wave_info = find_wr_waves_v4(wr, lookback=60, debug=debug)
    wave_info.update({
        "wave_complete":   wc["ok"],
        "deep_low_val":    wc["deep_low_val"],
        "bars_from_low":   wc["bars_from_low"],
        "peak_count":      wc["peak_count"],
    })

    buy_signals = []

    if wc["ok"]:
        buy_signals.append({
            "id":   "WAVE_COMPLETE",
            "name": "威廉波浪完成",
            "color": "#FF6B6B",
            "desc": (f"深底 WR={wc['deep_low_val']:.1f}（{wc['bars_from_low']}天前），"
                     f"形成 {wc['peak_count']} 個波峰，"
                     f"今日 WR={wc['wr_now']:.1f} 脫離超賣向上"),
            "priority": 5,
        })

    wr_raw  = wr.dropna().iloc[-30:]
    wr_list = [round(float(v), 2) for v in wr_raw.tolist()]

    return {
        "buy_signals": buy_signals,
        "wave_info":   wave_info,
        "wr_now":      round(float(sl[-1]), 2),
        "wr_list":     wr_list,
    }


# ──────────────────────────────────────────────────────────
# 獨立測試
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    test_stocks = [
        ("2455.TW", "全新"),
        ("2355.TW", "敬鵬"),
        ("8081.TW", "致新"),
        ("2313.TW", "華通"),
    ]
    print("=" * 60)
    print("  威廉波浪選股 v4（嚴格版）測試")
    print("=" * 60)

    for ticker, name in test_stocks:
        try:
            df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            if len(df) < 30:
                print(f"\n  {name}({ticker}): 資料不足"); continue

            result = calc_williams_signals_v4(df["Close"], df["High"], df["Low"])
            wi    = result["wave_info"]
            sigs  = result["buy_signals"]
            hit   = bool(sigs)
            mark  = "✓ 命中" if hit else "× 未命中"
            print(f"\n  {name}({ticker})  {mark}")
            print(f"    WR={result['wr_now']:.1f}  "
                  f"深底={wi.get('deep_low_val',0):.1f} "
                  f"距今{wi.get('bars_from_low',0)}天 "
                  f"波峰{wi.get('peak_count',0)}個")
            if sigs:
                for s in sigs:
                    print(f"    [{s['id']}] {s['name']}：{s['desc']}")
        except Exception as e:
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
