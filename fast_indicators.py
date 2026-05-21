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
    except Exception:
        pass
