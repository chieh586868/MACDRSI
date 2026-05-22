"""
================================================================
fast_indicators.py — numba JIT 加速的技術指標熱迴圈
================================================================
給 macd_app.py / divergence_screener.py 共用。
首次呼叫會編譯（約 1-3 秒），之後加速 10-50 倍。
"""
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def deco(f): return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco


# ════════════════════════════════════════════════════════════
#  UT Bot 追蹤止損線（取代原本 Python for-loop，加速 30x+）
# ════════════════════════════════════════════════════════════
@njit(cache=True, fastmath=True)
def ut_bot_trail(closes: np.ndarray, loss_thr: np.ndarray) -> np.ndarray:
    n = closes.shape[0]
    trail = np.empty(n, dtype=np.float64)
    if n == 0:
        return trail
    trail[0] = closes[0]
    for i in range(1, n):
        c  = closes[i]
        lt = loss_thr[i]
        pt = trail[i-1]
        if c > pt:
            v = c - lt
            trail[i] = v if v > pt else pt
        elif c < pt:
            v = c + lt
            trail[i] = v if v < pt else pt
        else:
            trail[i] = pt
    return trail


@njit(cache=True)
def ut_bot_trend(closes: np.ndarray, trail: np.ndarray) -> np.ndarray:
    n = closes.shape[0]
    out = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if closes[i] > trail[i]:
            out[i] = 1
        elif closes[i] < trail[i]:
            out[i] = -1
        else:
            out[i] = out[i-1]
    return out


# ════════════════════════════════════════════════════════════
#  Pivot 低/高點偵測（取代 Python 雙重迴圈，加速 20x+）
# ════════════════════════════════════════════════════════════
@njit(cache=True)
def find_pivot_lows_nb(values: np.ndarray, left: int, right: int) -> np.ndarray:
    n = values.shape[0]
    out = np.empty(n, dtype=np.int64)
    cnt = 0
    last = -10**9
    min_gap = left if left > 3 else 3
    for i in range(left, n - right):
        v = values[i]
        if v != v:  # NaN
            continue
        ok = True
        for j in range(1, left + 1):
            if v > values[i - j]:
                ok = False
                break
        if not ok:
            continue
        for j in range(1, right + 1):
            if v > values[i + j]:
                ok = False
                break
        if not ok:
            continue
        if i - last >= min_gap:
            out[cnt] = i
            cnt += 1
            last = i
    return out[:cnt]


@njit(cache=True)
def find_pivot_highs_nb(values: np.ndarray, left: int, right: int) -> np.ndarray:
    n = values.shape[0]
    out = np.empty(n, dtype=np.int64)
    cnt = 0
    last = -10**9
    min_gap = left if left > 3 else 3
    for i in range(left, n - right):
        v = values[i]
        if v != v:
            continue
        ok = True
        for j in range(1, left + 1):
            if v < values[i - j]:
                ok = False
                break
        if not ok:
            continue
        for j in range(1, right + 1):
            if v < values[i + j]:
                ok = False
                break
        if not ok:
            continue
        if i - last >= min_gap:
            out[cnt] = i
            cnt += 1
            last = i
    return out[:cnt]


# ════════════════════════════════════════════════════════════
#  Williams %R 波浪偵測（取代 find_wr_waves 內的 Python loop）
#  回傳 peaks 與 troughs 索引（已過濾相鄰同型）
# ════════════════════════════════════════════════════════════
@njit(cache=True)
def _raw_turns(values: np.ndarray, win: int):
    """回傳 (indices, kinds)：kinds 1=peak, -1=trough"""
    n = values.shape[0]
    idxs  = np.empty(n, dtype=np.int64)
    kinds = np.empty(n, dtype=np.int32)
    cnt = 0
    last_peak    = -10**9
    last_trough  = -10**9
    for i in range(win, n - win):
        v = values[i]
        # peak?
        is_peak = True
        lmax = values[i - 1]
        for k in range(2, win + 1):
            if values[i - k] > lmax:
                lmax = values[i - k]
        rmax = values[i + 1]
        for k in range(2, win + 1):
            if values[i + k] > rmax:
                rmax = values[i + k]
        if v >= lmax and v >= rmax:
            if i - last_peak >= 3:
                idxs[cnt]  = i
                kinds[cnt] = 1
                cnt += 1
                last_peak = i
            continue
        # trough?
        is_tr = True
        lmin = values[i - 1]
        for k in range(2, win + 1):
            if values[i - k] < lmin:
                lmin = values[i - k]
        rmin = values[i + 1]
        for k in range(2, win + 1):
            if values[i + k] < rmin:
                rmin = values[i + k]
        if v <= lmin and v <= rmin:
            if i - last_trough >= 3:
                idxs[cnt]  = i
                kinds[cnt] = -1
                cnt += 1
                last_trough = i
    return idxs[:cnt], kinds[:cnt]


def find_wr_turns(values: np.ndarray, win: int = 3, min_amp: float = 5.0):
    """
    回傳轉折點 list of (index, "peak"/"trough")，相鄰同型且差距 < min_amp 則合併。
    """
    idxs, kinds = _raw_turns(values, win)
    if len(idxs) == 0:
        return [], []
    filtered = []
    for i in range(len(idxs)):
        idx = int(idxs[i])
        kind = "peak" if kinds[i] == 1 else "trough"
        if filtered and filtered[-1][1] == kind:
            # 同型，保留更極端者
            prev_idx = filtered[-1][0]
            if (kind == "peak"   and values[idx] > values[prev_idx]) or \
               (kind == "trough" and values[idx] < values[prev_idx]):
                filtered[-1] = (idx, kind)
        else:
            if filtered and abs(values[idx] - values[filtered[-1][0]]) < min_amp:
                continue
            filtered.append((idx, kind))
    peaks   = [t[0] for t in filtered if t[1] == "peak"]
    troughs = [t[0] for t in filtered if t[1] == "trough"]
    return peaks, troughs


# ════════════════════════════════════════════════════════════
#  RSI / 滾動最大最小（取代 pandas rolling，加速 5-10x）
# ════════════════════════════════════════════════════════════
@njit(cache=True, fastmath=True)
def rolling_max_nb(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < window:
        return out
    for i in range(window - 1, n):
        m = x[i - window + 1]
        for k in range(i - window + 2, i + 1):
            if x[k] > m:
                m = x[k]
        out[i] = m
    return out


@njit(cache=True, fastmath=True)
def rolling_min_nb(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < window:
        return out
    for i in range(window - 1, n):
        m = x[i - window + 1]
        for k in range(i - window + 2, i + 1):
            if x[k] < m:
                m = x[k]
        out[i] = m
    return out


# ════════════════════════════════════════════════════════════
#  完整指標計算（取代 divergence_screener 的 calc_ind）
#  把 8 ewm + 3 rolling + 各種 series 運算合成單一 njit 函式
#  避免 pandas 每個 op 的 Python 層 ~1-2ms 開銷
#  預期加速 20-30x（calc_ind 從 ~50ms/股 → 1-2ms/股）
# ════════════════════════════════════════════════════════════
@njit(cache=True, fastmath=True)
def calc_indicators_fast(closes: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, volumes: np.ndarray):
    """
    回傳: (dif, dea, hist, rsi, K, D, ma5, ma10, ma20, vol_ratio)

    對應 divergence_screener 原版 pandas 計算：
      MACD: ewm(12), ewm(26), dif, dea=ewm(dif,9), hist=(dif-dea)*2
      RSI:  rolling(14) mean of gains/losses
      KD:   K=ewm(rsv, com=2), D=ewm(K, com=2)
      MA:   ewm(5), ewm(10), ewm(20)
      vol_ratio = volumes / rolling(5).mean(volumes)
    """
    n = closes.shape[0]

    dif       = np.empty(n)
    dea       = np.empty(n)
    hist      = np.empty(n)
    rsi       = np.full(n, np.nan)
    K         = np.empty(n)
    D         = np.empty(n)
    ma5       = np.empty(n)
    ma10      = np.empty(n)
    ma20      = np.empty(n)
    vol_ratio = np.ones(n)

    if n == 0:
        return dif, dea, hist, rsi, K, D, ma5, ma10, ma20, vol_ratio

    # MACD
    a12 = 2.0 / 13.0
    a26 = 2.0 / 27.0
    a9  = 2.0 / 10.0
    ema12 = closes[0]
    ema26 = closes[0]
    dea_v = 0.0
    dif[0]  = 0.0
    dea[0]  = 0.0
    hist[0] = 0.0
    for i in range(1, n):
        ema12 = a12 * closes[i] + (1.0 - a12) * ema12
        ema26 = a26 * closes[i] + (1.0 - a26) * ema26
        dif[i]  = ema12 - ema26
        dea_v   = a9 * dif[i] + (1.0 - a9) * dea_v
        dea[i]  = dea_v
        hist[i] = (dif[i] - dea_v) * 2.0

    # RSI(14)
    if n >= 15:
        g_sum = 0.0
        l_sum = 0.0
        for i in range(1, 15):
            d = closes[i] - closes[i-1]
            if d > 0:
                g_sum += d
            elif d < 0:
                l_sum -= d
        avg_l = l_sum / 14.0
        if avg_l == 0.0:
            rsi[14] = 100.0
        else:
            rs = (g_sum / 14.0) / avg_l
            rsi[14] = 100.0 - 100.0 / (1.0 + rs)
        for i in range(15, n):
            d_new = closes[i] - closes[i-1]
            d_old = closes[i-14] - closes[i-15]
            if d_new > 0:
                g_sum += d_new
            elif d_new < 0:
                l_sum -= d_new
            if d_old > 0:
                g_sum -= d_old
            elif d_old < 0:
                l_sum += d_old
            avg_l = l_sum / 14.0
            if avg_l == 0.0:
                rsi[i] = 100.0
            else:
                rs = (g_sum / 14.0) / avg_l
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    # KD(9)
    a_kd = 1.0 / 3.0
    K_v  = 50.0
    D_v  = 50.0
    for i in range(n):
        if i >= 8:
            lo = lows[i-8]
            hi = highs[i-8]
            for k in range(i-7, i+1):
                if lows[k]  < lo: lo = lows[k]
                if highs[k] > hi: hi = highs[k]
            denom = hi - lo + 1e-9
            rsv = (closes[i] - lo) / denom * 100.0
        else:
            rsv = 50.0
        K_v = a_kd * rsv + (1.0 - a_kd) * K_v
        D_v = a_kd * K_v + (1.0 - a_kd) * D_v
        K[i] = K_v
        D[i] = D_v

    # MA(5, 10, 20)
    a5  = 2.0 / 6.0
    a10 = 2.0 / 11.0
    a20 = 2.0 / 21.0
    m5  = closes[0]
    m10 = closes[0]
    m20 = closes[0]
    ma5[0]  = m5
    ma10[0] = m10
    ma20[0] = m20
    for i in range(1, n):
        m5  = a5  * closes[i] + (1.0 - a5)  * m5
        m10 = a10 * closes[i] + (1.0 - a10) * m10
        m20 = a20 * closes[i] + (1.0 - a20) * m20
        ma5[i]  = m5
        ma10[i] = m10
        ma20[i] = m20

    # Volume ratio
    if n >= 5:
        s = 0.0
        for i in range(5):
            s += volumes[i]
        vma = s / 5.0
        if vma > 0:
            vol_ratio[4] = volumes[4] / vma
        for i in range(5, n):
            s += volumes[i] - volumes[i-5]
            vma = s / 5.0
            if vma > 0:
                vol_ratio[i] = volumes[i] / vma

    return dif, dea, hist, rsi, K, D, ma5, ma10, ma20, vol_ratio


# ════════════════════════════════════════════════════════════
#  macd_app 用的完整指標版本（多出 OBV / Bollinger / Williams / MA60）
#  返回 18 個 numpy array
# ════════════════════════════════════════════════════════════
@njit(cache=True, fastmath=True)
def calc_all_indicators_fast(closes: np.ndarray, highs: np.ndarray,
                              lows: np.ndarray, volumes: np.ndarray):
    """
    返回 (dif, dea, hist, rsi, K, D,
          ma5, ma10, ma20, ma60,
          vol_ma5, vol_ratio,
          obv, obv_ma,
          bb_upper, bb_mid, bb_lower,
          wr)
    """
    n = closes.shape[0]

    dif       = np.empty(n)
    dea       = np.empty(n)
    hist      = np.empty(n)
    rsi       = np.full(n, np.nan)
    K         = np.empty(n)
    D         = np.empty(n)
    ma5       = np.empty(n)
    ma10      = np.empty(n)
    ma20      = np.empty(n)
    ma60      = np.empty(n)
    vol_ma5   = np.full(n, np.nan)
    vol_ratio = np.ones(n)
    obv       = np.zeros(n)
    obv_ma    = np.full(n, np.nan)
    bb_upper  = np.full(n, np.nan)
    bb_mid    = np.full(n, np.nan)
    bb_lower  = np.full(n, np.nan)
    wr        = np.full(n, np.nan)

    if n == 0:
        return (dif, dea, hist, rsi, K, D, ma5, ma10, ma20, ma60,
                vol_ma5, vol_ratio, obv, obv_ma,
                bb_upper, bb_mid, bb_lower, wr)

    # MACD
    a12 = 2.0/13.0; a26 = 2.0/27.0; a9 = 2.0/10.0
    ema12 = closes[0]; ema26 = closes[0]; dea_v = 0.0
    dif[0] = 0.0; dea[0] = 0.0; hist[0] = 0.0
    for i in range(1, n):
        ema12 = a12*closes[i] + (1.0-a12)*ema12
        ema26 = a26*closes[i] + (1.0-a26)*ema26
        dif[i] = ema12 - ema26
        dea_v  = a9*dif[i] + (1.0-a9)*dea_v
        dea[i] = dea_v
        hist[i] = (dif[i] - dea_v) * 2.0

    # RSI(14)
    if n >= 15:
        g_sum = 0.0; l_sum = 0.0
        for i in range(1, 15):
            d = closes[i] - closes[i-1]
            if d > 0: g_sum += d
            elif d < 0: l_sum -= d
        avg_l = l_sum / 14.0
        rsi[14] = 100.0 if avg_l == 0.0 else 100.0 - 100.0/(1.0 + (g_sum/14.0)/avg_l)
        for i in range(15, n):
            d_new = closes[i] - closes[i-1]
            d_old = closes[i-14] - closes[i-15]
            if d_new > 0: g_sum += d_new
            elif d_new < 0: l_sum -= d_new
            if d_old > 0: g_sum -= d_old
            elif d_old < 0: l_sum += d_old
            avg_l = l_sum / 14.0
            rsi[i] = 100.0 if avg_l == 0.0 else 100.0 - 100.0/(1.0 + (g_sum/14.0)/avg_l)

    # KD(9)
    a_kd = 1.0/3.0
    K_v = 50.0; D_v = 50.0
    for i in range(n):
        if i >= 8:
            lo = lows[i-8]; hi = highs[i-8]
            for k in range(i-7, i+1):
                if lows[k]  < lo: lo = lows[k]
                if highs[k] > hi: hi = highs[k]
            rsv = (closes[i] - lo) / (hi - lo + 1e-9) * 100.0
        else:
            rsv = 50.0
        K_v = a_kd*rsv + (1.0-a_kd)*K_v
        D_v = a_kd*K_v + (1.0-a_kd)*D_v
        K[i] = K_v; D[i] = D_v

    # MA(5/10/20/60)
    a5  = 2.0/6.0;  a10 = 2.0/11.0
    a20 = 2.0/21.0; a60 = 2.0/61.0
    m5  = closes[0]; m10 = closes[0]
    m20 = closes[0]; m60 = closes[0]
    ma5[0] = m5; ma10[0] = m10; ma20[0] = m20; ma60[0] = m60
    for i in range(1, n):
        m5  = a5 *closes[i] + (1.0-a5) *m5
        m10 = a10*closes[i] + (1.0-a10)*m10
        m20 = a20*closes[i] + (1.0-a20)*m20
        m60 = a60*closes[i] + (1.0-a60)*m60
        ma5[i] = m5; ma10[i] = m10; ma20[i] = m20; ma60[i] = m60

    # Volume MA(5) + ratio
    if n >= 5:
        s = 0.0
        for i in range(5):
            s += volumes[i]
        vma = s/5.0
        vol_ma5[4] = vma
        if vma > 0: vol_ratio[4] = volumes[4] / vma
        for i in range(5, n):
            s += volumes[i] - volumes[i-5]
            vma = s/5.0
            vol_ma5[i] = vma
            if vma > 0: vol_ratio[i] = volumes[i] / vma

    # OBV (cumsum of signed volume)
    cum = 0.0
    prev = closes[0]
    obv[0] = 0.0
    for i in range(1, n):
        c = closes[i]
        if c > prev:   cum += volumes[i]
        elif c < prev: cum -= volumes[i]
        obv[i] = cum
        prev = c

    # OBV MA(10)
    if n >= 10:
        s = 0.0
        for i in range(10):
            s += obv[i]
        obv_ma[9] = s/10.0
        for i in range(10, n):
            s += obv[i] - obv[i-10]
            obv_ma[i] = s/10.0

    # Bollinger(20, 2) — rolling mean + sample std (ddof=1) 配合 pandas 預設
    if n >= 20:
        for i in range(19, n):
            s = 0.0
            for k in range(i-19, i+1):
                s += closes[k]
            mean = s / 20.0
            ss = 0.0
            for k in range(i-19, i+1):
                d = closes[k] - mean
                ss += d * d
            std = (ss / 19.0) ** 0.5
            bb_mid[i]   = mean
            bb_upper[i] = mean + 2.0*std
            bb_lower[i] = mean - 2.0*std

    # Williams %R(20)
    if n >= 20:
        for i in range(19, n):
            hi = highs[i-19]; lo = lows[i-19]
            for k in range(i-18, i+1):
                if highs[k] > hi: hi = highs[k]
                if lows[k]  < lo: lo = lows[k]
            wr[i] = (hi - closes[i]) / (hi - lo + 1e-9) * -100.0

    return (dif, dea, hist, rsi, K, D, ma5, ma10, ma20, ma60,
            vol_ma5, vol_ratio, obv, obv_ma,
            bb_upper, bb_mid, bb_lower, wr)


# ════════════════════════════════════════════════════════════
#  預熱：啟動時呼叫一次強迫編譯，避免第一次 API 呼叫被卡 1-2 秒
# ════════════════════════════════════════════════════════════
def warmup():
    if not HAS_NUMBA:
        return
    try:
        x = np.linspace(100.0, 110.0, 80)
        lt = np.full(80, 1.0)
        t = ut_bot_trail(x, lt)
        _ = ut_bot_trend(x, t)
        _ = find_pivot_lows_nb(x, 3, 3)
        _ = find_pivot_highs_nb(x, 3, 3)
        _ = find_wr_turns(x, 3, 5.0)
        _ = rolling_max_nb(x, 5)
        _ = rolling_min_nb(x, 5)
        v = np.full(80, 1000.0)
        _ = calc_indicators_fast(x, x + 1, x - 1, v)
        _ = calc_all_indicators_fast(x, x + 1, x - 1, v)  # ★ 預編譯
    except Exception:
        pass
