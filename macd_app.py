"""
================================================================
MACD 盤中即時選股系統 - Flask 後端（極速版 v2）
================================================================
基於原速度優化版進一步加速：
  ★ 新增 1：numba JIT 編譯 UT Bot trail / pivot 偵測（10-50x）
  ★ 新增 2：eval_strategies / calc_confirm_score 預擷取 numpy（去除 .iloc 開銷）
  ★ 新增 3：analyze_stock 移除 pd.concat 單列附加（改用 numpy append）
  ★ 新增 4：fetch_history_batch 批次平行而非分組序列
  ★ 新增 5：batch_scan_all 預先批次下載所有歷史，再丟給 worker
  ★ 新增 6：sanitize 改非遞迴，並啟動時 numba 預編譯
================================================================
"""

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import json, time, threading, os
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
import yfinance as yf
from williams_v4 import calc_williams_signals_v4

# ── 速度優化：numba JIT 共用函式 ──────────────────────────
from fast_indicators import (
    ut_bot_trail, ut_bot_trend,
    find_pivot_lows_nb, find_pivot_highs_nb,
    calc_all_indicators_fast,
    warmup as fast_warmup,
)

app = Flask(__name__)
CORS(app)

app.config["JSON_SORT_KEYS"] = False
try:
    app.json.ensure_ascii = False
except Exception:
    pass


def sanitize(obj):
    """非遞迴清洗（用 stack 取代遞迴，省下大量 call frame 開銷）"""
    if isinstance(obj, (dict, list)):
        # 走訪所有節點原地修改
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if isinstance(v, float):
                        if v != v or v == float("inf") or v == float("-inf"):
                            cur[k] = 0.0
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            else:  # list
                for i, v in enumerate(cur):
                    if isinstance(v, float):
                        if v != v or v == float("inf") or v == float("-inf"):
                            cur[i] = 0.0
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
        return obj
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return 0.0
    return obj

# ══════════════════════════════════════════════════════════
# ▌ 全局常數與設定
# ══════════════════════════════════════════════════════════
HIST_CACHE_FILE = "hist_cache.pkl"
SCAN_CACHE_FILE = "scan_cache.pkl"
FULL_LIST_FILE  = "tw_stocks.json"
WATCHLIST2_FILE = "watchlist2.json"
INDUSTRY_FILE   = "industry_map.json"

YF_BATCH_SIZE   = 50
SCAN_WORKERS    = 40
YF_PARALLEL_BATCHES = 4         # ★ 新增：同組內幾批可同時下載
SJ_KBARS_SEM    = threading.Semaphore(5)

_global_executor = ThreadPoolExecutor(max_workers=SCAN_WORKERS)
# ★ 新增：給 yfinance 批次下載用的獨立 Pool，避免阻塞 scan workers
_yf_executor = ThreadPoolExecutor(max_workers=YF_PARALLEL_BATCHES * 2)

# ══════════════════════════════════════════════════════════
# ▌ 股票清單
# ══════════════════════════════════════════════════════════
DEFAULT_WATCHLIST = [
    {"id": "2330", "name": "台積電",   "ex": "tse"},
    {"id": "2317", "name": "鴻海",     "ex": "tse"},
    {"id": "2454", "name": "聯發科",   "ex": "tse"},
    {"id": "2881", "name": "富邦金",   "ex": "tse"},
    {"id": "2882", "name": "國泰金",   "ex": "tse"},
    {"id": "2891", "name": "中信金",   "ex": "tse"},
    {"id": "2303", "name": "聯電",     "ex": "tse"},
    {"id": "3711", "name": "日月光",   "ex": "tse"},
    {"id": "2382", "name": "廣達",     "ex": "tse"},
    {"id": "2308", "name": "台達電",   "ex": "tse"},
    {"id": "2412", "name": "中華電",   "ex": "tse"},
    {"id": "2002", "name": "中鋼",     "ex": "tse"},
]

full_stock_list  = []
full_list_loaded = False
full_list_lock   = threading.Lock()
watchlist        = list(DEFAULT_WATCHLIST)
cache            = {}
cache_lock       = threading.Lock()

# ══════════════════════════════════════════════════════════
# ▌ 歷史快取
# ══════════════════════════════════════════════════════════
hist_cache      = {}
hist_cache_lock = threading.Lock()
HIST_TTL        = 86400 * 3

def _hist_key(stock_id: str, period: str, ex: str) -> str:
    return f"{stock_id}_{period}_{ex}"

def _get_hist_cache(key: str):
    with hist_cache_lock:
        entry = hist_cache.get(key)
    if entry and (time.time() - entry["ts"]) < HIST_TTL:
        return entry["df"]
    return None

def _set_hist_cache(key: str, df: pd.DataFrame):
    with hist_cache_lock:
        hist_cache[key] = {"df": df, "ts": time.time()}


# ══════════════════════════════════════════════════════════
# ▌ yfinance 批量下載（★ 改為批次平行，TSE/OTC + 多批同時）
# ══════════════════════════════════════════════════════════
def fetch_history_batch(stocks: list, period: str = "3mo") -> dict:
    need_download = []
    result = {}

    for s in stocks:
        sid = s["id"]
        ex  = s.get("ex", "tse")
        key = _hist_key(sid, period, ex)
        cached_df = _get_hist_cache(key)
        if cached_df is not None and len(cached_df) >= 10:
            result[sid] = cached_df
        else:
            need_download.append(s)

    if not need_download:
        return result

    def _download_one_batch(batch, suffix, ex_name):
        sub = {}
        tickers_str = " ".join(f"{s['id']}{suffix}" for s in batch)
        try:
            raw = yf.download(
                tickers_str, period=period, group_by="ticker",
                progress=False, auto_adjust=True, threads=True,
            )
            single = len(batch) == 1
            cols0 = None if single else raw.columns.get_level_values(0)
            for s in batch:
                sid = s["id"]
                ticker = f"{sid}{suffix}"
                try:
                    if single:
                        df = raw
                    elif ticker in cols0:
                        df = raw[ticker]
                    else:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna()
                    if len(df) >= 10:
                        sub[sid] = df
                        _set_hist_cache(_hist_key(sid, period, ex_name), df)
                except Exception:
                    pass
        except Exception as e:
            # 批次失敗時改單支
            for s in batch:
                sid = s["id"]
                if sid in sub:
                    continue
                try:
                    df = yf.download(f"{sid}{suffix}", period=period,
                                     progress=False, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna()
                    if len(df) >= 10:
                        sub[sid] = df
                        _set_hist_cache(_hist_key(sid, period, ex_name), df)
                except Exception:
                    pass
        return sub

    # ★ 全部批次（TSE+OTC 混合）一起平行
    tasks = []
    tse_stocks = [s for s in need_download if s.get("ex", "tse") == "tse"]
    otc_stocks = [s for s in need_download if s.get("ex", "tse") == "otc"]
    for batch_start in range(0, len(tse_stocks), YF_BATCH_SIZE):
        tasks.append((tse_stocks[batch_start:batch_start + YF_BATCH_SIZE], ".TW", "tse"))
    for batch_start in range(0, len(otc_stocks), YF_BATCH_SIZE):
        tasks.append((otc_stocks[batch_start:batch_start + YF_BATCH_SIZE], ".TWO", "otc"))

    futures = [_yf_executor.submit(_download_one_batch, b, suf, ex) for b, suf, ex in tasks]
    for f in as_completed(futures):
        try:
            result.update(f.result())
        except Exception:
            pass
    return result


# ★ 下市/無資料黑名單
_delisted_set: set = set()
_delisted_lock = threading.Lock()
DELISTED_FILE = "delisted_cache.json"

def _load_delisted():
    global _delisted_set
    try:
        if os.path.exists(DELISTED_FILE):
            with open(DELISTED_FILE, "r") as f:
                _delisted_set = set(json.load(f))
            print(f"  📋 已知下市/無資料股票：{len(_delisted_set)} 支（略過）")
    except Exception:
        pass

def _save_delisted():
    try:
        with _delisted_lock:
            data = list(_delisted_set)
        with open(DELISTED_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def _mark_delisted(sid: str):
    with _delisted_lock:
        _delisted_set.add(sid)


def fetch_history(stock_id: str, period: str = "3mo", ex: str = "tse") -> pd.DataFrame:
    with _delisted_lock:
        if stock_id in _delisted_set:
            return pd.DataFrame()

    key = _hist_key(stock_id, period, ex)
    cached_df = _get_hist_cache(key)
    if cached_df is not None and len(cached_df) >= 10:
        return cached_df

    suffixes = [".TWO", ".TW"] if ex == "otc" else [".TW", ".TWO"]
    for suffix in suffixes:
        try:
            df = yf.download(f"{stock_id}{suffix}", period=period,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) >= 10:
                _set_hist_cache(key, df)
                return df
        except Exception:
            continue

    _mark_delisted(stock_id)
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# ▌ 啟動時批量預熱快取
# ══════════════════════════════════════════════════════════
preload_done = False
preload_lock = threading.Lock()

def warm_up_cache():
    global preload_done
    with preload_lock:
        if preload_done:
            return

    if _load_hist_cache_disk():
        preload_done = True
        print(f"  ✅ 歷史快取從磁碟載入完成（{len(hist_cache)} 檔）")
        return

    stocks = list(watchlist)
    total  = len(stocks)
    print(f"\n  🔄 批量預熱歷史快取（{total} 支）...")
    t0 = time.time()

    downloaded = fetch_history_batch(stocks, period="3mo")
    elapsed = round(time.time() - t0, 1)
    print(f"  ✅ 批量下載完成：{len(downloaded)}/{total} 支，耗時 {elapsed} 秒")

    _save_hist_cache_disk()
    preload_done = True


def _save_hist_cache_disk():
    try:
        import pickle
        with hist_cache_lock:
            data = dict(hist_cache)
        # ★ 用 protocol=4 + 較高效的二進位
        with open(HIST_CACHE_FILE, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  💾 歷史快取存檔（{len(data)} 檔）")
    except Exception as e:
        print(f"  ⚠ 快取存檔失敗: {e}")


def _load_hist_cache_disk() -> bool:
    global hist_cache
    try:
        import pickle
        if not os.path.exists(HIST_CACHE_FILE):
            return False
        mtime = os.path.getmtime(HIST_CACHE_FILE)
        if (time.time() - mtime) >= HIST_TTL:
            return False
        with open(HIST_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if len(data) < 100:
            return False
        with hist_cache_lock:
            hist_cache.update(data)
        return True
    except Exception as e:
        print(f"  ⚠ 快取載入失敗: {e}")
        return False


# ══════════════════════════════════════════════════════════
# ▌ scan_cache 持久化（避免每次重啟都要 141 秒重算）
#   TTL 6 小時，且要比 hist_cache.pkl 新
# ══════════════════════════════════════════════════════════
SCAN_CACHE_TTL = 6 * 3600  # 6 小時

def _save_scan_cache_disk():
    """把當前掃描結果存盤；只在 cache 有實質內容時存"""
    try:
        import pickle
        with cache_lock:
            data = dict(cache)
        if len(data) < 100:
            return
        with open(SCAN_CACHE_FILE, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  💾 掃描快取存檔（{len(data)} 檔）")
    except Exception as e:
        print(f"  ⚠ 掃描快取存檔失敗: {e}")


def _load_scan_cache_disk() -> bool:
    """啟動時嘗試載入；過期或比歷史快取舊則放棄"""
    global cache
    try:
        import pickle
        if not os.path.exists(SCAN_CACHE_FILE):
            return False
        scan_mtime = os.path.getmtime(SCAN_CACHE_FILE)
        if (time.time() - scan_mtime) >= SCAN_CACHE_TTL:
            return False
        # 若歷史快取比掃描快取新，掃描快取已過期
        if os.path.exists(HIST_CACHE_FILE):
            hist_mtime = os.path.getmtime(HIST_CACHE_FILE)
            if hist_mtime > scan_mtime + 60:  # 60 秒容差
                print(f"  ℹ 掃描快取早於歷史快取，重新計算")
                return False
        with open(SCAN_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if len(data) < 100:
            return False
        with cache_lock:
            cache.update(data)
        print(f"  ✅ 掃描快取載入：{len(cache)} 筆（fast mode 即用）")
        return True
    except Exception as e:
        print(f"  ⚠ 掃描快取載入失敗: {e}")
        return False


# ══════════════════════════════════════════════════════════
# ▌ 背離歷史資料庫（滾動 7 天，給條件 A/D 撈「幾天前背離、今天確認」）
# ══════════════════════════════════════════════════════════
import sqlite3
DIVERGENCE_DB   = "divergence_history.db"
DIV_DB_MIN      = 5          # 背離分數 ≥ 5 才寫入
DIV_DB_DAYS     = 7          # 保留滾動 7 天
_div_db_lock    = threading.Lock()

def _init_div_db():
    try:
        with sqlite3.connect(DIVERGENCE_DB) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS divergence_history(
                sid TEXT, name TEXT, date TEXT,
                daily_div INTEGER, weekly_div INTEGER, close REAL,
                PRIMARY KEY(sid, date))""")
    except Exception as e:
        print(f"  ⚠ 背離DB初始化失敗: {e}")

def record_divergence(results: list, min_div: int = DIV_DB_MIN):
    """掃描完成後，把背離分數 ≥ min_div 的個股寫進 DB（同日覆寫），並清掉 7 天前的"""
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for r in results:
        d = (r.get("div") or {}).get("score", 0)
        if d >= min_div:
            rows.append((r.get("id"), r.get("name",""), today, int(d),
                         int(r.get("weekly_score", 0) or 0), float(r.get("close", 0) or 0)))
    if not rows:
        return 0
    cutoff = (datetime.now() - timedelta(days=DIV_DB_DAYS)).strftime("%Y-%m-%d")
    try:
        with _div_db_lock, sqlite3.connect(DIVERGENCE_DB) as con:
            con.executemany("""INSERT OR REPLACE INTO divergence_history
                (sid,name,date,daily_div,weekly_div,close) VALUES (?,?,?,?,?,?)""", rows)
            con.execute("DELETE FROM divergence_history WHERE date < ?", (cutoff,))
        return len(rows)
    except Exception as e:
        print(f"  ⚠ 背離DB寫入失敗: {e}")
        return 0

def get_recent_divergence(days: int = DIV_DB_DAYS, min_div: int = DIV_DB_MIN) -> dict:
    """回傳 {sid: {div, last_date}}：過去 days 天內背離 ≥ min_div 的個股（取最高分）"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        with _div_db_lock, sqlite3.connect(DIVERGENCE_DB) as con:
            cur = con.execute("""SELECT sid, MAX(daily_div), MAX(date)
                FROM divergence_history WHERE date >= ? AND daily_div >= ?
                GROUP BY sid""", (cutoff, min_div))
            return {row[0]: {"div": row[1], "last_date": row[2]} for row in cur.fetchall()}
    except Exception as e:
        print(f"  ⚠ 背離DB讀取失敗: {e}")
        return {}


# ══════════════════════════════════════════════════════════
# ▌ 一次計算所有技術指標（★ 全 numba JIT）
# ══════════════════════════════════════════════════════════
def calc_all_indicators(closes: pd.Series, highs: pd.Series,
                         lows: pd.Series, volumes: pd.Series) -> dict:
    """
    ★ 完全用 numba JIT 計算（含 OBV / Bollinger / Williams / MA60 等）
    比原 pandas 版本快約 20-30 倍。輸出仍包成 pd.Series 維持下游相容。
    UT Bot 因依賴前一根狀態，使用獨立 numba 函式分開算。
    """
    n   = len(closes)
    idx = closes.index

    # ── numpy 陣列化 ──────────────────────────────────
    c_arr = np.asarray(closes, dtype=np.float64)
    h_arr = np.asarray(highs,  dtype=np.float64)
    l_arr = np.asarray(lows,   dtype=np.float64)
    v_arr = np.asarray(volumes, dtype=np.float64)

    # ── 主指標（單一 numba 呼叫一次算完）────────────
    (dif, dea, hist, rsi, K, D,
     ma5, ma10, ma20, ma60,
     vol_ma5, vol_ratio, obv, obv_ma,
     bb_upper, bb_mid, bb_lower, wr) = calc_all_indicators_fast(
        c_arr, h_arr, l_arr, v_arr
    )

    ind = {
        "dif":       pd.Series(dif,       index=idx),
        "dea":       pd.Series(dea,       index=idx),
        "hist":      pd.Series(hist,      index=idx),
        "rsi":       pd.Series(rsi,       index=idx),
        "K":         pd.Series(K,         index=idx),
        "D":         pd.Series(D,         index=idx),
        "ma5":       pd.Series(ma5,       index=idx),
        "ma10":      pd.Series(ma10,      index=idx),
        "ma20":      pd.Series(ma20,      index=idx),
        "ma60":      pd.Series(ma60,      index=idx),
        "vol_ma5":   pd.Series(vol_ma5,   index=idx),
        "vol_ratio": pd.Series(vol_ratio, index=idx),
        "obv":       pd.Series(obv,       index=idx),
        "obv_ma":    pd.Series(obv_ma,    index=idx),
        "bb_upper":  pd.Series(bb_upper,  index=idx),
        "bb_mid":    pd.Series(bb_mid,    index=idx),
        "bb_lower":  pd.Series(bb_lower,  index=idx),
        "wr":        pd.Series(wr,        index=idx),
    }

    # ── UT Bot（用 numba ut_bot_trail / ut_bot_trend）
    try:
        # atr_period=1 + ewm(span=1) → alpha=1 → atr = tr (無平滑)
        prev_close = np.empty(n)
        prev_close[0]  = c_arr[0]
        prev_close[1:] = c_arr[:-1]
        tr_arr = np.maximum(
            np.maximum(h_arr - l_arr, np.abs(h_arr - prev_close)),
            np.abs(l_arr - prev_close)
        )
        loss_thr = 3.0 * tr_arr  # key_value=3.0
        trail_arr = ut_bot_trail(c_arr, loss_thr)
        ut_t      = ut_bot_trend(c_arr, trail_arr)
        # ut_buy/ut_sell 用 numpy 算交叉
        prev_c    = np.empty(n); prev_c[0]    = c_arr[0];    prev_c[1:]    = c_arr[:-1]
        prev_trl  = np.empty(n); prev_trl[0]  = trail_arr[0]; prev_trl[1:] = trail_arr[:-1]
        ut_buy_arr  = (c_arr > trail_arr) & (prev_c <= prev_trl)
        ut_sell_arr = (c_arr < trail_arr) & (prev_c >= prev_trl)
        ind["ut_trail"] = pd.Series(trail_arr,   index=idx)
        ind["ut_buy"]   = pd.Series(ut_buy_arr,  index=idx)
        ind["ut_sell"]  = pd.Series(ut_sell_arr, index=idx)
        ind["ut_trend"] = pd.Series(ut_t,        index=idx)
    except Exception:
        ind["ut_trail"] = pd.Series(np.zeros(n), index=idx)
        ind["ut_buy"]   = pd.Series(False, index=idx)
        ind["ut_sell"]  = pd.Series(False, index=idx)
        ind["ut_trend"] = pd.Series(0, index=idx)

    return ind


# ══════════════════════════════════════════════════════════
# ▌ 兼容用單一指標函式
# ══════════════════════════════════════════════════════════
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(closes, fast=12, slow=26, signal=9):
    ef = closes.ewm(span=fast, adjust=False).mean()
    es = closes.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2

def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_kd(highs, lows, closes, period=9, k_smooth=3, d_smooth=3):
    low_n  = lows.rolling(period).min()
    high_n = highs.rolling(period).max()
    rsv = (closes - low_n) / (high_n - low_n + 1e-9) * 100
    K   = rsv.ewm(com=k_smooth-1, adjust=False).mean()
    D   = K.ewm(com=d_smooth-1,   adjust=False).mean()
    return K, D

def calc_obv(closes, volumes):
    return (np.sign(closes.diff()).fillna(0) * volumes).cumsum()

def calc_bollinger(closes, period=20, std_mult=2):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid + std * std_mult, mid, mid - std * std_mult

def calc_williams_r(highs, lows, closes, period=20):
    h = highs.rolling(period).max()
    l = lows.rolling(period).min()
    return (h - closes) / (h - l + 1e-9) * -100


# ══════════════════════════════════════════════════════════
# ▌ 策略評估（★ 預擷取 numpy 純量，避免重複 .iloc）
# ══════════════════════════════════════════════════════════
def _last_scalar(s, default=0.0):
    try:
        v = s.values[-1]
        return default if v != v else float(v)
    except Exception:
        return default

def _prev_scalar(s, default=0.0):
    """倒數第二個值（1 日前）"""
    try:
        v = s.values[-2]
        return default if v != v else float(v)
    except Exception:
        return default

def _recent3_high(s, default=0.0):
    """近3日最高：2日前/3日前/4日前 三日最高（排除今日與昨日，避免與昨收比較失去意義）"""
    try:
        vals = [float(v) for v in s.values[-5:-2] if v == v]
        return max(vals) if vals else default
    except Exception:
        return default

def eval_strategies(ind: dict, closes: pd.Series, volumes: pd.Series) -> list:
    if len(closes) < 3:
        return []

    # ★ 一次性把所有需要的尾端純量轉成 float
    dif_v  = ind["dif"].values
    dea_v  = ind["dea"].values
    hist_v = ind["hist"].values
    c_v    = closes.values
    d0, d1 = float(dif_v[-1]),  float(dif_v[-2])
    e0, e1 = float(dea_v[-1]),  float(dea_v[-2])
    h0, h1 = float(hist_v[-1]), float(hist_v[-2])
    h2     = float(hist_v[-3]) if len(hist_v) > 2 else h1
    c0     = float(c_v[-1])

    ma5  = _last_scalar(ind["ma5"])
    ma10 = _last_scalar(ind["ma10"])
    ma20 = _last_scalar(ind["ma20"])
    ma60 = _last_scalar(ind["ma60"], ma20) if "ma60" in ind else ma20

    vr_v = ind["vol_ratio"].values
    vol_ratio = float(vr_v[-1]) if vr_v[-1] == vr_v[-1] else 1.0
    vol_ok    = vol_ratio >= 1.2

    golden = (d0 > e0) and (d1 <= e1)
    dead   = (d0 < e0) and (d1 >= e1)
    above_zero = d0 > 0 and e0 > 0
    below_zero = d0 < 0 and e0 < 0

    rsi_v = ind["rsi"].values
    rsi_now = float(rsi_v[-1]) if rsi_v[-1] == rsi_v[-1] else 50.0

    if len(c_v) >= 65:
        ma60_v = ind["ma60"].values
        ma60_slope = float(ma60_v[-1] - ma60_v[-5])
    else:
        ma60_slope = 0.0

    triggered = []

    # S1 底部黃金交叉
    if golden and below_zero and h0 > 0 and vol_ok and 25 <= rsi_now <= 55 \
       and (ma60_slope >= 0 or c0 > ma60 * 0.95):
        triggered.append({"id": "S1", "name": "底部黃金交叉", "color": "#FFD700", "priority": 4,
            "desc": f"零軸下DIF上穿DEA，量比{vol_ratio:.1f}x RSI={rsi_now:.0f}"})

    # S2 零軸動能擴張
    if above_zero and h0 > 0 and h0 > h1 and c0 > ma10 and c0 > ma20 \
       and rsi_now < 68 and ma60_slope > 0 and vol_ratio >= 1.0 and c0 > ma60:
        triggered.append({"id": "S2", "name": "零軸動能擴張", "color": "#00E5FF", "priority": 2,
            "desc": f"紅柱擴張+MA60向上，RSI={rsi_now:.0f}，站MA20={ma20:.1f}"})

    # S3 底背離
    if len(c_v) >= 30:
        nw = 30
        prev_low_p = float(c_v[-nw:].min())
        prev_low_d = float(dif_v[-nw:].min())
        if c0 <= prev_low_p * 1.002 and d0 > prev_low_d * 0.9 and d0 < 0 and golden:
            triggered.append({"id": "S3", "name": "底背離反轉", "color": "#CE93D8", "priority": 3,
                "desc": "價格新低但MACD未創低，已黃金交叉確認"})

    # S4 多週期共振
    weekly_bull = above_zero
    if weekly_bull and golden and vol_ok:
        triggered.append({"id": "S4", "name": "多週期共振", "color": "#FF1744", "priority": 5,
            "desc": f"週線+日線同向多頭，量比{vol_ratio:.1f}x"})

    # S5 柱縮量進場
    shrink1 = h1 > 0 and h1 < h2
    expand  = h0 > 0 and h0 > h1
    high5   = float(c_v[-6:-1].max()) if len(c_v) >= 6 else c0
    if shrink1 and expand and c0 > ma20 and d0 > e0 \
       and vol_ratio >= 1.5 and c0 >= high5 * 0.995 and rsi_now >= 45 and c0 > ma60:
        triggered.append({"id": "S5", "name": "柱縮量進場", "color": "#69F0AE", "priority": 2,
            "desc": f"紅柱縮後放量{vol_ratio:.1f}x突破前高，RSI={rsi_now:.0f}"})

    # S6 強勢股回測
    if len(c_v) >= 20:
        prev_high = float(c_v[-20:-1].max())
        breakout  = float(c_v[-6:-1].max()) > prev_high * 0.98
        pullback  = ma10 * 0.97 <= c0 <= ma10 * 1.02
        if breakout and pullback and not dead and above_zero and h0 > 0:
            triggered.append({"id": "S6", "name": "強勢股回測", "color": "#FF9800", "priority": 3,
                "desc": f"突破後回踩MA10={ma10:.1f}，MACD未死叉"})

    return sorted(triggered, key=lambda x: -x["priority"])


def calc_confirm_score(ind: dict, closes: pd.Series, volumes: pd.Series) -> dict:
    score   = 0
    details = {}
    try:
        c_v   = closes.values
        c0    = float(c_v[-1])
        ma5   = _last_scalar(ind["ma5"])
        ma10  = _last_scalar(ind["ma10"])
        ma20  = _last_scalar(ind["ma20"])
        ma60  = _last_scalar(ind["ma60"], ma20) if len(c_v) >= 60 else ma20

        # 均線
        if ma5 > ma10 > ma20 > ma60:
            score += 2
            details["均線"] = {"status": "完整多頭排列", "score": 2, "pass": True,
                               "detail": f"MA5{ma5:.1f}>MA10{ma10:.1f}>MA20{ma20:.1f}>MA60{ma60:.1f}"}
        elif ma5 > ma10 > ma20:
            score += 1
            details["均線"] = {"status": "短期多頭排列", "score": 1, "pass": True,
                               "detail": f"MA5{ma5:.1f}>MA10{ma10:.1f}>MA20{ma20:.1f}"}
        else:
            details["均線"] = {"status": "尚未排列", "score": 0, "pass": False, "detail": ""}

        # KD
        K_v = ind["K"].values; D_v = ind["D"].values
        k0, d0_ = float(K_v[-1]), float(D_v[-1])
        k1, d1_ = float(K_v[-2]), float(D_v[-2])
        if k0 > d0_ and k1 <= d1_ and k0 < 30:
            score += 2
            details["KD"] = {"status": "超賣黃金交叉", "score": 2, "pass": True,
                              "detail": f"K={k0:.1f} D={d0_:.1f}"}
        elif k0 > d0_ and k1 <= d1_:
            score += 1
            details["KD"] = {"status": "黃金交叉", "score": 1, "pass": True,
                              "detail": f"K={k0:.1f} D={d0_:.1f}"}
        elif k0 > d0_ and k0 > 50:
            score += 1
            details["KD"] = {"status": "多方強勢", "score": 1, "pass": True,
                              "detail": f"K={k0:.1f}"}
        else:
            details["KD"] = {"status": "偏弱", "score": 0, "pass": False,
                              "detail": f"K={k0:.1f} D={d0_:.1f}"}

        # RSI
        rsi_v = ind["rsi"].values
        r0, r1 = float(rsi_v[-1]), float(rsi_v[-2])
        if r0 == r0:
            if r0 > 50 and r1 <= 50 and 50 < r0 < 70:
                score += 2
                details["RSI"] = {"status": "突破50強勢", "score": 2, "pass": True,
                                   "detail": f"RSI={r0:.1f}"}
            elif 30 <= r0 < 70:
                score += 1
                details["RSI"] = {"status": "位置適當", "score": 1, "pass": True,
                                   "detail": f"RSI={r0:.1f}"}
            elif r0 >= 70:
                details["RSI"] = {"status": "過熱警示", "score": 0, "pass": False,
                                   "detail": f"RSI={r0:.1f}"}
            else:
                details["RSI"] = {"status": "偏弱", "score": 0, "pass": False,
                                   "detail": f"RSI={r0:.1f}"}

        # OBV
        obv_v   = ind["obv"].values
        obv_ma_v= ind["obv_ma"].values
        obv0    = float(obv_v[-1])
        obv_ma0 = float(obv_ma_v[-1])
        obv_max = float(obv_v[-20:].max())
        if obv0 >= obv_max * 0.98 and obv0 > obv_ma0:
            score += 2
            details["OBV"] = {"status": "創高放量", "score": 2, "pass": True,
                               "detail": "OBV突破近期高點"}
        elif obv0 > obv_ma0:
            score += 1
            details["OBV"] = {"status": "量能健康", "score": 1, "pass": True,
                               "detail": "OBV站上均線"}
        else:
            details["OBV"] = {"status": "量能偏弱", "score": 0, "pass": False, "detail": ""}

        # 布林
        bu_v = ind["bb_upper"].values
        bm_v = ind["bb_mid"].values
        bl_v = ind["bb_lower"].values
        bu0, bm0, bl0 = float(bu_v[-1]), float(bm_v[-1]), float(bl_v[-1])
        if bm0 == bm0:
            bw_now  = (bu0 - bl0) / bm0 if bm0 > 0 else 0
            bm5     = float(bm_v[-5]) if len(c_v) >= 5 else bm0
            bu5     = float(bu_v[-5]) if len(c_v) >= 5 else bu0
            bl5     = float(bl_v[-5]) if len(c_v) >= 5 else bl0
            bw_prev = (bu5 - bl5) / bm5 if bm5 > 0 else 0
            expanding = bw_now > bw_prev and c0 > bm0
            if expanding:
                score += 2
                details["布林"] = {"status": "開口擴張多方", "score": 2, "pass": True,
                                    "detail": f"站上中軌{bm0:.1f}，通道擴張"}
            elif c0 > bm0:
                score += 1
                details["布林"] = {"status": "站上中軌", "score": 1, "pass": True,
                                    "detail": f"中軌={bm0:.1f}"}
            elif c0 <= bl0 * 1.02:
                score += 1
                details["布林"] = {"status": "下軌支撐", "score": 1, "pass": True,
                                    "detail": f"靠近下軌{bl0:.1f}"}
            else:
                details["布林"] = {"status": "中軌下方", "score": 0, "pass": False, "detail": ""}

    except Exception:
        pass

    if score >= 9:   grade, gc = "🔥 強烈買進", "#FF1744"
    elif score >= 7: grade, gc = "⭐ 積極買進", "#FF9800"
    elif score >= 5: grade, gc = "👀 觀察追蹤", "#FFD700"
    elif score >= 3: grade, gc = "⏳ 繼續等待", "#78909C"
    else:            grade, gc = "❌ 條件不足", "#546E7A"

    return {"score": score, "max_score": 10, "grade": grade, "grade_color": gc,
            "details": details,
            "kd_k": _last_scalar(ind.get("K",   pd.Series([0]))),
            "kd_d": _last_scalar(ind.get("D",   pd.Series([0]))),
            "rsi":  _last_scalar(ind.get("rsi", pd.Series([0])))}


# ══════════════════════════════════════════════════════════
# ▌ Pivot 偵測（★ 改用 numba JIT）
# ══════════════════════════════════════════════════════════
def find_pivot_lows(series: pd.Series, left: int = 3, right: int = 3) -> list:
    arr = series.values.astype(np.float64)
    return find_pivot_lows_nb(arr, left, right).tolist()

def find_pivot_highs(series: pd.Series, left: int = 3, right: int = 3) -> list:
    arr = series.values.astype(np.float64)
    return find_pivot_highs_nb(arr, left, right).tolist()


def detect_divergence(price, indicator, pivot_left=3, pivot_right=3,
                       lookback=120, min_gap=4) -> dict:
    result = {"regular_bull": False, "hidden_bull": False,
              "regular_bars": 999,   "hidden_bars": 999,
              "price_low1": 0.0, "price_low2": 0.0,
              "ind_low1": 0.0,   "ind_low2": 0.0, "detail": ""}
    min_len = pivot_left + pivot_right + min_gap + 5
    if len(price) < min_len:
        return result
    actual_lb = min(lookback, len(price))
    p_sl = price.iloc[-actual_lb:].ffill().bfill().fillna(0)
    i_sl = indicator.iloc[-actual_lb:].ffill().bfill().fillna(0)
    pv = p_sl.values.astype(np.float64)
    iv = i_sl.values.astype(np.float64)
    pp = find_pivot_lows_nb(pv, pivot_left, pivot_right)
    ip = find_pivot_lows_nb(iv, pivot_left, pivot_right)
    if len(pp) < 2 or len(ip) < 2:
        return result
    n  = len(pv)
    p1_idx = int(pp[-1])
    p2_idx = None
    for x in reversed(pp[:-1].tolist()):
        if p1_idx - x >= min_gap:
            p2_idx = int(x); break
    if p2_idx is None:
        return result
    sr = max(pivot_left + pivot_right, 6)
    ip_list = ip.tolist()
    def nearest_ip(pidx):
        best = None
        for x in ip_list:
            if abs(x - pidx) <= sr:
                if best is None or abs(x - pidx) < abs(best - pidx):
                    best = x
        return best if best is not None else pidx
    i1_idx = nearest_ip(p1_idx); i2_idx = nearest_ip(p2_idx)
    p1, p2 = float(pv[p1_idx]), float(pv[p2_idx])
    i1, i2 = float(iv[i1_idx]), float(iv[i2_idx])
    if any(v != v for v in [p1, p2, i1, i2]):
        return result
    bars_since = n - 1 - p1_idx
    if p1 < p2 and i1 > i2:
        result["regular_bull"]  = True
        result["regular_bars"]  = bars_since
        result["detail"]        = f"正規底背離：價格{p2:.2f}→{p1:.2f}，指標底部抬高，距今{bars_since}根"
    if p1 > p2 and i1 < i2:
        result["hidden_bull"]   = True
        result["hidden_bars"]   = bars_since
        result["detail"]        = f"隱藏底背離：價格更高低點，指標更低，距今{bars_since}根"
    result.update({"price_low1": round(p1,2), "price_low2": round(p2,2),
                   "ind_low1": round(i1,4),   "ind_low2": round(i2,4)})
    return result


def calc_divergence_signals(ind: dict, closes: pd.Series, highs: pd.Series,
                              lows: pd.Series, volumes: pd.Series,
                              pivot_left=3, pivot_right=3,
                              lookback=120, max_bars=15) -> dict:
    buy_signals = []; score = 0

    try:
        md = detect_divergence(closes, ind["hist"], pivot_left, pivot_right, lookback)
        if md["regular_bull"] and md["regular_bars"] <= max_bars:
            score += 2
            buy_signals.append({"id": "DIV_MACD_R", "name": "MACD 正規底背離",
                "color": "#CE93D8", "desc": md["detail"]+"（反轉訊號）",
                "type": "regular", "indicator": "MACD", "priority": 4})
        if md["hidden_bull"] and md["hidden_bars"] <= max_bars:
            score += 1
            buy_signals.append({"id": "DIV_MACD_H", "name": "MACD 隱藏底背離",
                "color": "#80DEEA", "desc": md["detail"]+"（趨勢延續）",
                "type": "hidden", "indicator": "MACD", "priority": 3})
    except Exception:
        md = {}

    try:
        rd = detect_divergence(closes, ind["rsi"], pivot_left, pivot_right, lookback)
        if rd["regular_bull"] and rd["regular_bars"] <= max_bars:
            score += 2
            buy_signals.append({"id": "DIV_RSI_R", "name": "RSI 正規底背離",
                "color": "#A5D6A7", "desc": rd["detail"]+"（反轉訊號）",
                "type": "regular", "indicator": "RSI", "priority": 4})
        if rd["hidden_bull"] and rd["hidden_bars"] <= max_bars:
            score += 1
            buy_signals.append({"id": "DIV_RSI_H", "name": "RSI 隱藏底背離",
                "color": "#80CBC4", "desc": rd["detail"]+"（趨勢延續）",
                "type": "hidden", "indicator": "RSI", "priority": 3})
    except Exception:
        rd = {}

    try:
        kd = detect_divergence(closes, ind["K"], pivot_left, pivot_right, lookback)
        if kd["regular_bull"] and kd["regular_bars"] <= max_bars:
            score += 2
            buy_signals.append({"id": "DIV_KD_R", "name": "KD 正規底背離",
                "color": "#FFCC80", "desc": kd["detail"]+"（反轉訊號）",
                "type": "regular", "indicator": "KD", "priority": 3})
        if kd["hidden_bull"] and kd["hidden_bars"] <= max_bars:
            score += 1
            buy_signals.append({"id": "DIV_KD_H", "name": "KD 隱藏底背離",
                "color": "#FFAB91", "desc": kd["detail"]+"（趨勢延續）",
                "type": "hidden", "indicator": "KD", "priority": 2})
    except Exception:
        kd = {}

    reg_c = sum(1 for s in buy_signals if s["type"] == "regular")
    hid_c = sum(1 for s in buy_signals if s["type"] == "hidden")
    if reg_c >= 2: score += 1
    if hid_c >= 2: score += 1

    parts = []
    if reg_c: parts.append(f"正規底背離：{'+'.join(s['indicator'] for s in buy_signals if s['type']=='regular')}")
    if hid_c: parts.append(f"隱藏底背離：{'+'.join(s['indicator'] for s in buy_signals if s['type']=='hidden')}")

    buy_signals.sort(key=lambda x: -x["priority"])
    return {"buy_signals": buy_signals, "macd_div": md, "rsi_div": rd, "kd_div": kd,
            "score": min(score, 6), "summary": "；".join(parts) if parts else "無背離訊號",
            "regular_count": reg_c, "hidden_count": hid_c}


# ══════════════════════════════════════════════════════════
# ▌ 費波南 / 威廉
# ══════════════════════════════════════════════════════════
def calc_fibonacci_signals(closes, opens, highs, lows, ind: dict) -> dict:
    n = len(closes)
    if n < 25:
        return {"basic_pass": False, "signals_count": 0, "buy_signals": [],
                "details": {}, "fib_levels": {}, "pressure_bars": []}
    c = closes.values; o = opens.values; h = highs.values; l = lows.values
    ma3  = float(c[-3:].mean())
    ma5  = _last_scalar(ind["ma5"])
    ma10 = _last_scalar(ind["ma10"])
    c0 = float(c[-1]); c1 = float(c[-2]); h2 = float(h[-3])
    basic_pass = (c1 < h2) and (c1 < ma5 or c1 < ma10)

    lookback_days = [1, 2, 3, 4, 5, 8, 13]
    pressure_bars = []
    for d in lookback_days:
        if n <= d + 1: continue
        idx  = -(d + 1); ck = float(c[idx]); ok = float(o[idx])
        pressure = ok if ck >= ok else (ck + ok) / 2
        pressure_bars.append({"days_ago": d, "is_red": ck >= ok,
                               "pressure": round(pressure, 2), "broken": c0 > pressure})
    broken_count   = sum(1 for pb in pressure_bars if pb["broken"])
    above_ma3_or_5 = c0 > ma3 or c0 > ma5

    week_signals = 0
    def gp(days_ago):
        idx = -(days_ago + 1)
        if abs(idx) > n: return None
        ck = float(c[idx]); ok = float(o[idx])
        return ok if ck >= ok else (ck + ok) / 2
    for da, lbl in [(5,"週1"),(1,"週1K"),(4,"週4")]:
        p = gp(da)
        if p and c0 > p: week_signals += 1

    def sm_body(idx):
        if abs(idx) > n-1: return False
        body = abs(float(c[idx])-float(o[idx])); rng = float(h[idx])-float(l[idx])
        return rng > 0 and body/rng < 0.3
    double_star = sm_body(-2) and sm_body(-3)
    def is_harami():
        if n < 3: return False
        ph = max(float(o[-3]),float(c[-3])); pl = min(float(o[-3]),float(c[-3]))
        ch = max(float(o[-2]),float(c[-2])); cl = min(float(o[-2]),float(c[-2]))
        return ph > ch and pl < cl
    harami = is_harami()
    pattern_found = double_star or harami

    lb_high = min(50, n-1)
    rb_idx  = n - lb_high + int(highs.iloc[-lb_high:].values.argmax())
    bars_fh = n - 1 - rb_idx
    fib_bar_levels = [5, 8, 13, 21]
    fib_bar_hit    = [fb for fb in fib_bar_levels if abs(bars_fh - fb) <= 1]
    sw_high = float(highs.iloc[-lb_high:].max())
    sw_low  = float(lows.iloc[-lb_high:].min())
    pr      = sw_high - sw_low
    fib_supports = {"0.618": round(sw_high - pr*0.618,2),
                    "0.500": round(sw_high - pr*0.5,2),
                    "0.382": round(sw_high - pr*0.382,2)}
    near_fib = any(abs(c0-v)/v < 0.02 for v in fib_supports.values())

    buy_signals = []
    if above_ma3_or_5 and broken_count >= 4:
        buy_signals.append({"id":"FA","name":"費波南A：4訊號突破",
            "desc":f"突破{broken_count}個變盤壓力","color":"#FF6B35"})
    if week_signals >= 2 and broken_count >= 3:
        buy_signals.append({"id":"FB","name":"費波南B：週期突破",
            "desc":f"週期訊號{week_signals}個","color":"#FF9F1C"})
    if pattern_found and above_ma3_or_5 and broken_count >= 3:
        pname = "雙星" if double_star else "孕育K"
        buy_signals.append({"id":"FC","name":f"費波南C：{pname}+突破",
            "desc":f"{pname}型態+{broken_count}個訊號","color":"#FFBF69"})
    if fib_bar_hit and above_ma3_or_5 and broken_count >= 3:
        buy_signals.append({"id":"FD","name":"費波南D：數列壓力突破",
            "desc":f"距高點{bars_fh}根，突破壓力","color":"#CBF3F0"})
    if near_fib and above_ma3_or_5:
        buy_signals.append({"id":"FE","name":"費波南E：黃金分割支撐",
            "desc":"近黃金分割支撐","color":"#2EC4B6"})

    return {"basic_pass": basic_pass, "signals_count": broken_count,
            "buy_signals": buy_signals,
            "details": {"broken_count": broken_count, "above_ma3_or_5": above_ma3_or_5,
                        "week_signals": week_signals, "pattern_found": pattern_found,
                        "bars_from_high": bars_fh, "fib_bar_hit": fib_bar_hit,
                        "near_fib_support": near_fib,
                        "ma3": round(ma3, 2), "ma5": round(ma5, 2), "ma10": round(ma10, 2),
                        "close": round(c0, 2)},
            "fib_levels": fib_supports, "pressure_bars": pressure_bars}


def calc_williams_signals(ind: dict, closes: pd.Series,
                           highs: pd.Series = None, lows: pd.Series = None) -> dict:
    h = highs if highs is not None else closes
    l = lows  if lows  is not None else closes
    return calc_williams_signals_v4(closes, h, l, ind=ind)


# ══════════════════════════════════════════════════════════
# ▌ 永豐 Shioaji
# ══════════════════════════════════════════════════════════
try:
    from config import SHIOAJI_API_KEY, SHIOAJI_SECRET_KEY
except ImportError:
    SHIOAJI_API_KEY = ""; SHIOAJI_SECRET_KEY = ""

try:
    import shioaji as sj
    HAS_SJ = True
except ImportError:
    HAS_SJ = False

_sj_api = None; _sj_lock = threading.Lock(); _sj_ready = False

def init_shioaji():
    global _sj_api, _sj_ready
    if not HAS_SJ or not SHIOAJI_API_KEY:
        print("⚠ 未設定永豐 API 金鑰，使用 TWSE 延遲報價"); return False
    try:
        with _sj_lock:
            if _sj_ready: return True
            api = sj.Shioaji(simulation=False)
            api.login(api_key=SHIOAJI_API_KEY, secret_key=SHIOAJI_SECRET_KEY, fetch_contract=True)
            _sj_api = api; _sj_ready = True
            print("✅ 永豐 Shioaji 連線成功"); return True
    except Exception as e:
        print(f"❌ 永豐連線失敗：{e}"); return False

def fetch_shioaji_batch(stock_ids: list) -> dict:
    now = datetime.now()
    t   = now.hour * 60 + now.minute
    is_trading = (9*60 <= t <= 13*60+35) and (now.weekday() < 5)
    if not is_trading:
        return {}
    if not _sj_ready or _sj_api is None: return {}
    result = {}
    try:
        contracts = []; id_map = {}
        for sid in stock_ids:
            for market in [_sj_api.Contracts.Stocks.TSE, _sj_api.Contracts.Stocks.OTC]:
                try:
                    c = market[sid]
                    if c: contracts.append(c); id_map[c.code] = sid; break
                except: continue
        if not contracts: return {}
        snaps = _sj_api.snapshots(contracts)
        for snap in snaps:
            try:
                sid = id_map.get(snap.code, snap.code)
                close = float(snap.close) if snap.close else 0
                if close <= 0: continue
                chg_p = float(getattr(snap,"change_price",0) or 0)
                result[sid] = {"close": close,
                    "open":   float(snap.open)  if snap.open  else close,
                    "high":   float(snap.high)  if snap.high  else close,
                    "low":    float(snap.low)   if snap.low   else close,
                    "yest":   close - chg_p,
                    "volume": float(getattr(snap,"total_volume",None) or snap.volume or 0),
                    "name":   getattr(snap,"name",sid),
                    "time":   str(getattr(snap,"ts",""))[:19], "source": "shioaji"}
            except: continue
    except Exception as e:
        print(f"  批次報價錯誤: {e}")
    return result

TWSE_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
HEADERS  = {"User-Agent":"Mozilla/5.0","Referer":"https://mis.twse.com.tw/"}

def fetch_twse_price(stock_id, ex="tse"):
    try:
        key = f"{ex}_{stock_id}.tw"
        r = requests.get(TWSE_URL, params={"ex_ch":key,"json":1,"delay":0},
                         headers=HEADERS, timeout=5)
        item = r.json().get("msgArray",[{}])[0]
        if not item.get("z") or item["z"]=="-": return None
        return {"close":float(item.get("z",0)),"open":float(item.get("o",0)),
                "high":float(item.get("h",0)),"low":float(item.get("l",0)),
                "yest":float(item.get("y",0)),"volume":float(item.get("v",0)),
                "name":item.get("n",stock_id),"time":item.get("t",""),"source":"twse"}
    except: return None

def fetch_yahoo_today(stock_id, ex="tse"):
    try:
        suffix = ".TW" if ex=="tse" else ".TWO"
        info = yf.Ticker(f"{stock_id}{suffix}").fast_info
        close = float(info.last_price) if info.last_price else 0
        if close == 0: return None
        prev = float(info.previous_close) if info.previous_close else close
        return {"close":close,"open":float(info.open or close),
                "high":float(info.day_high or close),"low":float(info.day_low or close),
                "yest":prev,"volume":float(info.three_month_average_volume or 0),
                "name":stock_id,"time":datetime.now().strftime("%H:%M:%S"),"source":"yahoo"}
    except: return None

_rt_batch_cache = {}; _rt_batch_time = 0; _rt_batch_lock = threading.Lock(); _RT_BATCH_TTL = 60

def fetch_realtime_price(stock_id, ex="tse"):
    with _rt_batch_lock:
        if time.time() - _rt_batch_time < _RT_BATCH_TTL:
            rt = _rt_batch_cache.get(stock_id)
            if rt and rt.get("close",0) > 0: return rt
    # 非交易時段不逐支連網：冷啟動時逐支打 Shioaji/TWSE/Yahoo 會全部 timeout，
    # 把整體掃描拖到 30 分鐘。休市時直接回 None，改用歷史最後一根 K 線。
    now = datetime.now()
    t_min = now.hour * 60 + now.minute
    is_trading = (9*60 <= t_min <= 13*60+35) and (now.weekday() < 5)
    if not is_trading:
        return None
    if _sj_ready:
        rt = fetch_shioaji_batch([stock_id]).get(stock_id)
        if rt and rt.get("close",0) > 0: return rt
    rt = fetch_twse_price(stock_id, ex)
    if rt and rt.get("close",0) > 0: return rt
    return fetch_yahoo_today(stock_id, ex)

threading.Thread(target=init_shioaji, daemon=True).start()


# ══════════════════════════════════════════════════════════
# ▌ analyze_stock（★ 移除 pd.concat 單列附加）
# ══════════════════════════════════════════════════════════
def _append_one(series: pd.Series, value, new_idx) -> pd.Series:
    """以 numpy append 取代 pd.concat 單列附加，加速 ~5-10x"""
    arr = np.append(series.values, value)
    return pd.Series(arr, index=series.index.append(pd.DatetimeIndex([new_idx])))


def analyze_stock(stock: dict) -> dict:
    sid = stock["id"]; ex = stock.get("ex","tse")
    hist = fetch_history(sid, ex=ex)
    has_history = len(hist) >= 30
    rt = fetch_realtime_price(sid, ex)

    def clean(v, d=0):
        try:
            f = float(v)
            return d if (f != f or np.isinf(f)) else f
        except: return d

    if has_history:
        # 取出原始 Series（不額外 copy，後面只在需要時才寫入）
        closes  = hist["Close"]
        # yfinance 成交量是「股」，台股慣用「張」(1張=1000股)。即時報價(Shioaji/TWSE)
        # 已是「張」，故在此把歷史量 ÷1000 統一為「張」，避免單位混用導致 vol_ratio 失真。
        volumes = hist["Volume"] / 1000.0
        highs   = hist["High"]  if "High"  in hist.columns else closes
        lows_s  = hist["Low"]   if "Low"   in hist.columns else closes
        opens_s = hist["Open"]  if "Open"  in hist.columns else closes

        # 更新今日即時報價
        if rt and rt.get("close", 0) > 0:
            today = pd.Timestamp(date.today())
            if today in closes.index:
                # 已存在今日 → 直接覆寫（避免 SettingWithCopyWarning：先 copy 再寫）
                closes  = closes.copy();  closes.iloc[-1]  = rt["close"]
                volumes = volumes.copy(); volumes.iloc[-1] = rt.get("volume", volumes.iloc[-1])
                highs   = highs.copy();   highs.iloc[-1]   = rt.get("high",   highs.iloc[-1])
                lows_s  = lows_s.copy();  lows_s.iloc[-1]  = rt.get("low",    lows_s.iloc[-1])
            else:
                # 用 numpy append（比 pd.concat 快 5x+）
                closes  = _append_one(closes,  rt["close"], today)
                volumes = _append_one(volumes, rt.get("volume", volumes.values[-1]), today)
                highs   = _append_one(highs,   rt.get("high",  rt["close"]),         today)
                lows_s  = _append_one(lows_s,  rt.get("low",   rt["close"]),         today)
                opens_s = _append_one(opens_s, rt.get("open",  rt["close"]),         today)

        ind = calc_all_indicators(closes, highs, lows_s, volumes)

        signals  = eval_strategies(ind, closes, volumes)
        confirm  = calc_confirm_score(ind, closes, volumes)
        fib_res  = calc_fibonacci_signals(closes, opens_s, highs, lows_s, ind)
        wr_res   = calc_williams_signals_v4(closes, highs, lows_s, ind=ind)
        try:
            div_res = calc_divergence_signals(ind, closes, highs, lows_s, volumes,
                                                lookback=120, max_bars=15)
        except Exception:
            div_res = {"buy_signals":[],"score":0,"summary":"","macd_div":{},"rsi_div":{},
                       "kd_div":{},"regular_count":0,"hidden_count":0}

        ut_buy_v   = ind["ut_buy"].values
        ut_sell_v  = ind["ut_sell"].values
        ut_trend_v = ind["ut_trend"].values
        ut_trail_v = ind["ut_trail"].values
        ut_buy_now   = bool(ut_buy_v[-1])
        ut_sell_now  = bool(ut_sell_v[-1])
        ut_trend_now = int(ut_trend_v[-1])
        ut_trail_now = round(float(ut_trail_v[-1]), 2)

        vr_v = ind["vol_ratio"].values
        vol_ratio = clean(vr_v[-1], 1.0)
        close_now = clean(closes.values[-1])
        yest      = rt.get("yest", clean(closes.values[-2])) if rt else clean(closes.values[-2])
        sparkline = [clean(v) for v in closes.values[-30:].tolist()]

    else:
        ind = {}
        signals = []; confirm = {"score":0,"max_score":10,"grade":"資料不足",
            "grade_color":"#546E7A","details":{},"kd_k":0,"kd_d":0,"rsi":0}
        vol_ratio = 1.0; close_now = rt["close"] if rt else 0; yest = rt.get("yest",0) if rt else 0
        sparkline = []; ut_buy_now = ut_sell_now = False; ut_trend_now = 0; ut_trail_now = 0.0
        fib_res = {"basic_pass":False,"signals_count":0,"buy_signals":[],"details":{},"fib_levels":{},"pressure_bars":[]}
        wr_res  = {"buy_signals":[],"wave_info":{},"wr_now":0,"wr_list":[]}
        div_res = {"buy_signals":[],"score":0,"summary":"資料不足","macd_div":{},"rsi_div":{},"kd_div":{},"regular_count":0,"hidden_count":0}

    change = close_now - yest
    change_pct = (change / yest * 100) if yest > 0 else 0

    return {
        "id":           sid,
        "name":         next((s.get("name","") for s in watchlist if s["id"]==sid), "") or
                        (rt["name"] if rt else stock.get("name", sid)),
        "close":        clean(close_now),
        "yest":         clean(yest),
        "change":       clean(change),
        "change_pct":   clean(change_pct),
        "volume":       int(volumes.values[-1]) if has_history else (int(rt.get("volume",0)) if rt else 0),
        "vol_ratio":    clean(vol_ratio, 1.0),
        "ma5":  _last_scalar(ind.get("ma5",  pd.Series([0]))) if ind else 0,
        "ma10": _last_scalar(ind.get("ma10", pd.Series([0]))) if ind else 0,
        "ma20": _last_scalar(ind.get("ma20", pd.Series([0]))) if ind else 0,
        "ma60": _last_scalar(ind.get("ma60", pd.Series([0]))) if ind else 0,
        "prev_close": _prev_scalar(closes) if has_history else 0,
        "prev3_high": _recent3_high(highs) if has_history else 0,
        "prev_ma5":   _prev_scalar(ind.get("ma5",  pd.Series([0]))) if ind else 0,
        "prev_ma10":  _prev_scalar(ind.get("ma10", pd.Series([0]))) if ind else 0,
        "dif":  _last_scalar(ind.get("dif",  pd.Series([0]))) if ind else 0,
        "dea":  _last_scalar(ind.get("dea",  pd.Series([0]))) if ind else 0,
        "hist": _last_scalar(ind.get("hist", pd.Series([0]))) if ind else 0,
        "prev_dif":  _prev_scalar(ind.get("dif",  pd.Series([0]))) if ind else 0,
        "prev_dea":  _prev_scalar(ind.get("dea",  pd.Series([0]))) if ind else 0,
        "prev_hist": _prev_scalar(ind.get("hist", pd.Series([0]))) if ind else 0,
        "signals":      signals, "confirm": confirm,
        "ut_buy":  ut_buy_now,  "ut_sell": ut_sell_now,
        "ut_trend":ut_trend_now,"ut_trail":ut_trail_now,
        "ut_macd_combo": ut_buy_now, "ut_combo_strong": ut_buy_now,
        "fib":  fib_res, "wr": wr_res, "div": div_res,
        "sparkline":   sparkline,
        "hist_bars":   [],
        "rt_time":     rt["time"]   if rt else "",
        "rt_source":   rt.get("source","twse") if rt else "none",
        "has_data":    has_history,
        "scanned_at":  datetime.now().strftime("%H:%M:%S"),
    }

def analyze_stock_safe(stock: dict):
    try:
        return analyze_stock(stock)
    except Exception as e:
        print(f"  ⚠ {stock['id']} 分析失敗: {e}")
        return None


# ══════════════════════════════════════════════════════════
# ▌ 快速模式 analyze
# ══════════════════════════════════════════════════════════
def analyze_stock_fast(stock: dict):
    sid = stock["id"]; ex = stock.get("ex","tse")
    with _rt_batch_lock:
        rt = _rt_batch_cache.get(sid)
    with cache_lock:
        cached = cache.get(sid)

    if cached and cached.get("has_data") and "prev3_high" in cached and rt and rt.get("close",0) > 0:
        close_now = rt["close"]
        yest = rt.get("yest", cached.get("yest", close_now))
        change = close_now - yest
        change_pct = (change / yest * 100) if yest > 0 else 0
        updated = dict(cached)
        updated.update({"close": round(close_now,2), "change": round(change,2),
            "change_pct": round(change_pct,2), "rt_time": rt.get("time",""),
            "rt_source": rt.get("source",""), "scanned_at": datetime.now().strftime("%H:%M:%S"),
            "volume": int(rt.get("volume", cached.get("volume",0))),
            "name": next((s.get("name","") for s in watchlist if s["id"]==sid), cached.get("name",""))})
        return updated

    now = datetime.now()
    t_min = now.hour * 60 + now.minute
    is_trading = (9*60 <= t_min <= 13*60+35) and (now.weekday() < 5)
    # 休市時直接用 cache（不重算）；但若是舊版 cache（缺 prev_close 欄位）或
    # vol_ratio 異常為 0（舊版休市抓報價 volume=0 算壞），則強制重算修正。
    cache_is_fresh = (cached and "prev_close" in cached and "prev3_high" in cached
                      and cached.get("vol_ratio", 0) > 0)
    if cached and cached.get("has_data") and not is_trading and cache_is_fresh:
        updated = dict(cached)
        updated["scanned_at"] = datetime.now().strftime("%H:%M:%S")
        updated["rt_source"]  = "cache_offhour"
        return updated

    return analyze_stock_safe(stock)


# ══════════════════════════════════════════════════════════
# ▌ 主掃描（★ 先批量補齊歷史快取，再丟給 worker）
# ══════════════════════════════════════════════════════════
def batch_scan_all(batch_size: int = 100, delay: float = 0.0,
                   fast_mode: bool = True) -> list:
    stocks = list(watchlist); total = len(stocks)
    results = []; signals_found = 0

    if fast_mode and len(hist_cache) < 50:
        fast_mode = False

    scan_fn = analyze_stock_fast if fast_mode else analyze_stock_safe
    mode_str = "快速" if fast_mode else "完整"
    print(f"  📊 {mode_str}掃描：{total} 支，{SCAN_WORKERS} 線程")

    # ★ 完整模式：先把所有缺失歷史一次批次補齊
    if not fast_mode:
        missing = []
        for s in stocks:
            key = _hist_key(s["id"], "3mo", s.get("ex", "tse"))
            if _get_hist_cache(key) is None:
                with _delisted_lock:
                    if s["id"] in _delisted_set:
                        continue
                missing.append(s)
        if missing:
            print(f"  ⬇  批次補齊歷史：{len(missing)} 支...")
            t0 = time.time()
            fetch_history_batch(missing, period="3mo")
            print(f"  ✅ 歷史補齊 {round(time.time()-t0,1)} 秒")

    # 一次取得所有即時報價
    if _sj_ready:
        all_ids = [s["id"] for s in stocks]
        print("  🔄 批次取得即時報價...")
        t0 = time.time()
        batch_prices = fetch_shioaji_batch(all_ids)
        with _rt_batch_lock:
            global _rt_batch_cache, _rt_batch_time
            _rt_batch_cache = batch_prices; _rt_batch_time = time.time()
        print(f"  ✅ 報價取得 {len(batch_prices)}/{total} 支 ({round(time.time()-t0,1)}秒)")

    t0 = time.time()
    futures = {_global_executor.submit(scan_fn, s): s for s in stocks}

    done_count = 0
    for future in as_completed(futures):
        try:
            r = future.result(timeout=30)
            if r and r.get("close",0) > 0 and r.get("volume",0) >= 10:
                results.append(r)
                with cache_lock: cache[r["id"]] = r
                if (r.get("fib",{}).get("buy_signals") or
                        r.get("wr",{}).get("buy_signals") or r.get("ut_buy")):
                    signals_found += 1
        except Exception:
            pass
        done_count += 1
        if done_count % 100 == 0:
            pct = done_count * 100 // total
            print(f"  進度 {done_count}/{total} ({pct}%) 訊號{signals_found}檔", end="\r")

    elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ {mode_str}掃描完成：{len(results)}/{total} 支,{signals_found} 個訊號，{elapsed} 秒")

    # ★ 掃描結果存盤（不論 mode）。下次啟動可直接 fast mode
    #   存盤條件：結果 >= 100 筆才存（避免空跑）
    if len(results) >= 100:
        threading.Thread(target=_save_scan_cache_disk, daemon=True).start()

    return results


# ══════════════════════════════════════════════════════════
# ▌ 股票清單管理
# ══════════════════════════════════════════════════════════
def fetch_twse_stock_list():
    try:
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                         timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        return [{"id":str(x["Code"]),"name":str(x["Name"]),"ex":"tse"}
                for x in r.json() if str(x.get("Code","")).isdigit() and len(str(x["Code"]))==4]
    except Exception as e:
        print(f"  ⚠ TWSE清單失敗: {e}"); return []

def fetch_otc_stock_list():
    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
                         timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        return [{"id":str(x["SecuritiesCompanyCode"]),"name":str(x["CompanyName"]),"ex":"otc"}
                for x in r.json() if str(x.get("SecuritiesCompanyCode","")).isdigit()
                and len(str(x["SecuritiesCompanyCode"]))==4]
    except Exception as e:
        print(f"  ⚠ TPEx清單失敗: {e}"); return []

# ── 產業類別 map (sid -> 產業) ─────────────────────────────
industry_map = {}
industry_lock = threading.Lock()

# 預設排除產業關鍵字（用「包含」比對，不需要完整名稱一致）
DEFAULT_EXCLUDED_KEYWORDS = (
    "水泥","塑膠","紡織","建材","營造",
    "航運","海運","造紙","觀光","餐旅","橡膠"
)

def is_excluded_industry(ind_name: str) -> bool:
    if not ind_name: return False
    return any(k in ind_name for k in DEFAULT_EXCLUDED_KEYWORDS)

# 向下相容（其他地方還在用 set）
DEFAULT_EXCLUDED_INDUSTRIES = set()  # 已被 is_excluded_industry 取代

_INDUSTRY_CODE_MAP = {
    "01":"水泥工業","02":"食品工業","03":"塑膠工業","04":"紡織纖維",
    "05":"電機機械","06":"電器電纜","08":"玻璃陶瓷","09":"造紙工業",
    "10":"鋼鐵工業","11":"橡膠工業","12":"汽車工業","14":"建材營造",
    "15":"航運業","16":"觀光餐旅","17":"金融保險","18":"貿易百貨",
    "20":"其他","21":"化學工業","22":"生技醫療","23":"油電燃氣",
    "24":"半導體業","25":"電腦及週邊設備","26":"光電業","27":"通信網路",
    "28":"電子零組件","29":"電子通路","30":"資訊服務","31":"其他電子",
    "32":"文化創意","33":"農業科技","34":"電子商務",
}

def _normalize_industry(v: str) -> str:
    """純數字 1-2 位 → 用 code 對照；否則回原字串"""
    if not v: return ""
    s = str(v).strip()
    if s.isdigit() and len(s) <= 2:
        return _INDUSTRY_CODE_MAP.get(s.zfill(2), s)
    return s

def _extract_industry_field(x: dict) -> str:
    # 先試已知 key，再 fuzzy 找含「產業」或 "industry" 的欄位
    for k in ("產業類別","產業別","SubIndustryCategory","IndustryCategory",
             "Category","產業","SubIndustry","SecuritiesIndustryCode"):
        v = x.get(k)
        if v: return _normalize_industry(v)
    for k, v in x.items():
        kl = str(k).lower()
        if (("產業" in str(k)) or ("industry" in kl) or ("category" in kl)) and v:
            return _normalize_industry(v)
    return ""

def _extract_sid_field(x: dict) -> str:
    for k in ("公司代號","SecuritiesCompanyCode","CompanyCode","Code","股票代號"):
        v = x.get(k)
        if v: return str(v).strip()
    for k, v in x.items():
        kl = str(k).lower()
        if (("代號" in str(k)) or ("code" in kl)) and v:
            s = str(v).strip()
            if s.isdigit() and 3 <= len(s) <= 6: return s
    return ""

def fetch_industry_map():
    """從 TWSE / TPEx 拿產業類別，建 dict[sid -> 產業名稱]。7 天 cache。"""
    global industry_map
    print(f"  📂 載入產業 map...")
    if os.path.exists(INDUSTRY_FILE):
        try:
            age_days = (time.time() - os.path.getmtime(INDUSTRY_FILE)) / 86400
            if age_days < 7:
                with open(INDUSTRY_FILE,"r",encoding="utf-8") as f:
                    data = json.load(f)
                if len(data) > 100:
                    with industry_lock: industry_map = data
                    print(f"  ✅ 產業 map 從 cache 載入：{len(data)} 檔（{age_days:.1f} 天前）"); return
                print(f"  ⚠ cache 內容過少（{len(data)} 檔），重抓")
            else:
                print(f"  ⏰ cache 過期（{age_days:.1f} 天），重抓")
        except Exception as e:
            print(f"  ⚠ 讀 cache 失敗: {e}")
    out = {}
    # 上市
    try:
        url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        arr = r.json() if r.status_code == 200 else []
        n_before = len(out)
        for x in arr:
            sid = _extract_sid_field(x)
            ind = _extract_industry_field(x)
            if sid and ind: out[sid] = ind
        print(f"  · TWSE 產業 API: status={r.status_code}, 筆數={len(arr)}, 解析出 {len(out)-n_before} 檔")
        if arr:
            print(f"    所有 keys: {list(arr[0].keys())}")
            samples_3 = [(s, out.get(s,"(無)")) for s in ("1101","1525","2603","2330")]
            print(f"    抽樣 sid→ind: {samples_3}")
    except Exception as e:
        print(f"  ⚠ TWSE 產業資料失敗: {e}")
    # 上櫃
    try:
        url = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        arr = r.json() if r.status_code == 200 else []
        n_before = len(out)
        for x in arr:
            sid = _extract_sid_field(x)
            ind = _extract_industry_field(x)
            if sid and ind: out[sid] = ind
        print(f"  · TPEx 產業 API: status={r.status_code}, 筆數={len(arr)}, 解析出 {len(out)-n_before} 檔")
    except Exception as e:
        print(f"  ⚠ TPEx 產業資料失敗: {e}")
    if len(out) > 100:
        try:
            with open(INDUSTRY_FILE,"w",encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠ 寫 cache 失敗: {e}")
        with industry_lock: industry_map = out
        # 列出排除產業在 map 裡的數量 + 統計前 15 大產業類別
        excl_count = sum(1 for v in out.values() if is_excluded_industry(v))
        print(f"  ✅ 產業 map：{len(out)} 檔（預設排除產業命中 {excl_count} 檔）")
        from collections import Counter
        top = Counter(out.values()).most_common(15)
        print(f"  📊 產業分布 Top15: {top}")
    else:
        print(f"  ❌ 產業 map 抓取失敗，僅 {len(out)} 檔。產業過濾將失效（但 MA 緊纏過濾仍生效）")


def load_full_stock_list():
    global full_stock_list, full_list_loaded, watchlist
    if os.path.exists(FULL_LIST_FILE):
        try:
            if (time.time() - os.path.getmtime(FULL_LIST_FILE)) < 86400:
                with open(FULL_LIST_FILE,"r",encoding="utf-8") as f:
                    data = json.load(f)
                if len(data) > 100:
                    with full_list_lock:
                        full_stock_list = data; full_list_loaded = True; watchlist = data
                    print(f"  ✅ 從快取載入完整清單：{len(data)} 檔"); return
        except: pass
    tse = fetch_twse_stock_list(); otc = fetch_otc_stock_list()
    seen = set(); unique = []
    for s in tse + otc:
        if s["id"] not in seen: seen.add(s["id"]); unique.append(s)
    if len(unique) > 100:
        with open(FULL_LIST_FILE,"w",encoding="utf-8") as f:
            json.dump(unique, f, ensure_ascii=False)
        with full_list_lock:
            full_stock_list = unique; full_list_loaded = True; watchlist = unique
        print(f"  ✅ 完整清單載入：{len(unique)} 檔")

# ══════════════════════════════════════════════════════════
# ▌ 觀察清單 2
# ══════════════════════════════════════════════════════════
watchlist2 = []
watchlist2_lock = threading.Lock()

def load_watchlist2():
    global watchlist2
    try:
        if os.path.exists(WATCHLIST2_FILE):
            with open(WATCHLIST2_FILE,"r",encoding="utf-8") as f:
                watchlist2 = json.load(f)
    except: pass

def save_watchlist2(data):
    try:
        with open(WATCHLIST2_FILE,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"儲存失敗: {e}")

def add_to_watchlist2(r: dict, min_score: int = 3):
    global watchlist2
    score = (r.get("confirm") or {}).get("score", 0)
    if score < min_score or not r.get("signals"): return False
    sid = r.get("id"); name = r.get("name", sid)
    ex  = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
    with watchlist2_lock:
        existing = next((s for s in watchlist2 if s["id"]==sid), None)
        if existing:
            existing.update({"score":score,"signals":[s["id"] for s in r.get("signals",[])],
                "last_update":datetime.now().strftime("%Y-%m-%d %H:%M"),"close":r.get("close",0)})
            save_watchlist2(watchlist2); return False
        else:
            watchlist2.append({"id":sid,"name":name,"ex":ex,"score":score,
                "signals":[s["id"] for s in r.get("signals",[])],
                "close":r.get("close",0),"added_date":datetime.now().strftime("%Y-%m-%d"),
                "added_time":datetime.now().strftime("%H:%M"),
                "last_update":datetime.now().strftime("%Y-%m-%d %H:%M"),
                "ut_buy":False,"ut_trail":r.get("ut_trail",0),"note":""})
            save_watchlist2(watchlist2); return True

load_watchlist2()


# ══════════════════════════════════════════════════════════
# ▌ Telegram
# ══════════════════════════════════════════════════════════
def _load_telegram_config():
    """Telegram 設定來源優先序：環境變數 > 同資料夾 telegram_config.txt。
    設定檔每行 KEY=VALUE：
        TELEGRAM_TOKEN=機器人金鑰
        TELEGRAM_CHAT_ID=你的聊天室ID
    檔案放 macd_app.py 同資料夾，不進 git、curl 更新也不會被覆蓋 → 設一次永久有效。"""
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_config.txt")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip().upper(); v = v.strip().strip('"').strip("'")
                    if   k == "TELEGRAM_TOKEN"   and not token: token = v   # 環境變數優先
                    elif k == "TELEGRAM_CHAT_ID" and not chat:  chat  = v
    except Exception as e:
        print(f"⚠ 讀取 telegram_config.txt 失敗: {e}")
    return token, chat

TELEGRAM_TOKEN, TELEGRAM_CHAT_ID = _load_telegram_config()
TELEGRAM_API     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    print(f"✓ Telegram 已設定 (chat_id={TELEGRAM_CHAT_ID})")
else:
    print("⚠ Telegram 未設定：設環境變數 TELEGRAM_TOKEN/TELEGRAM_CHAT_ID，或在同資料夾放 telegram_config.txt")
telegram_enabled = True   # ★ UI 可動態開關（不需重啟）

def send_telegram(msg):
    if not telegram_enabled: return False
    if not TELEGRAM_TOKEN: return False
    try:
        r = requests.post(TELEGRAM_API, json={"chat_id":TELEGRAM_CHAT_ID,
            "text":msg,"parse_mode":"HTML"}, timeout=10)
        return r.status_code == 200
    except: return False


# 手動掃描完成時的去重快取（同 mode 5分鐘內不重複推送）
_manual_push_last = {}
_manual_push_lock = threading.Lock()

def push_scan_summary(mode: str, rows: list, max_show: int = 15):
    """掃描完成時把結果摘要推到 Telegram，5 分鐘內同 mode 不重發"""
    if not telegram_enabled or not TELEGRAM_TOKEN: return
    if not rows: return
    with _manual_push_lock:
        last = _manual_push_last.get(mode)
        if last and (datetime.now() - last).total_seconds() < 300:
            return
        _manual_push_last[mode] = datetime.now()
    mode_label = {"A":"UT+威廉+背離","B":"UT+費波南+週線威廉波浪","C+":"UT+費費+威廉+週MACD多頭",
                  "D":"日週雙重背離","E":"日週MACD共振(依進場優先分排序)",
                  "F":"週威廉波浪+ABCDE任一+昨日回檔(嚴格)",
                  "G":"(UT或費波南)+(日威廉/背離/MACD共振/週威廉)+昨日回檔+變盤<4"}.get(mode, mode)
    lines = [f"📊 <b>條件 {mode} 掃描完成</b>",
             f"🕐 {datetime.now().strftime('%H:%M:%S')}  找到 {len(rows)} 支",
             f"📋 {mode_label}", ""]
    if mode == "E":
        for i, r in enumerate(rows[:max_show], 1):
            c = r.get("change_pct", 0)
            arrow = "▲" if c > 0 else "▼"
            wk = r.get("weekly_macd") or {}
            wtag = "週零軸上" if wk.get("above_zero") else "週零軸下"
            if wk.get("golden_cross"): wtag += "剛金叉"
            div_s = (r.get("div") or {}).get("score", 0)
            dtag = f" 背離{div_s}" if div_s >= 2 else ""
            lines.append(f"{i}. <b>{r.get('name','')}({r.get('id','')})</b> "
                         f"${r.get('close',0):.2f} {arrow}{abs(c):.2f}%")
            lines.append(f"    優先{r.get('prio',0)}分 ｜ E{r.get('e_score',0)} ｜ "
                         f"{r.get('level_tag','')}(離MA20 {r.get('bias20','?')}%) ｜ {wtag}{dtag}")
    else:
        for i, r in enumerate(rows[:max_show], 1):
            c = r.get("change_pct", 0)
            arrow = "▲" if c > 0 else "▼"
            lines.append(f"{i}. <b>{r.get('name','')}({r.get('id','')})</b> "
                         f"${r.get('close',0):.2f} {arrow}{abs(c):.2f}% "
                         f"量{r.get('volume',0)}")
    if len(rows) > max_show:
        lines.append(f"...另 {len(rows)-max_show} 支")
    lines.append("\n⚠ 僅供參考")
    try:
        send_telegram("\n".join(lines))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# ▌ 定時排程
# ══════════════════════════════════════════════════════════
scheduler_running = False; scheduler_interval = 30; last_notified = {}

def is_market_hour():
    now = datetime.now()
    if now.weekday() >= 5: return False
    t = now.hour*60 + now.minute
    return 9*60 <= t <= 13*60+30

def auto_scan_job():
    t0 = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏱ 自動掃描（{len(watchlist)}支）...")
    results = batch_scan_all()
    if not results: return
    elapsed = round(time.time()-t0,1)
    try: record_divergence(results)
    except Exception: pass
    for r in results:
        try: add_to_watchlist2(r, min_score=3)
        except: pass
    notify_groups = {}
    for r in results:
        sid   = r.get("id")
        has_fib = bool(r.get("fib",{}).get("buy_signals"))
        has_ut  = bool(r.get("ut_buy"))
        has_wr  = bool(r.get("wr",{}).get("buy_signals"))
        div     = r.get("div",{})
        has_div = bool(div.get("buy_signals"))
        is_b = has_ut and has_fib and has_div
        is_a = has_ut and has_wr and has_div
        is_c = (has_fib or has_ut) and has_wr
        if not (is_b or is_a or is_c): continue
        lt = last_notified.get(sid)
        if lt and (datetime.now()-lt).seconds < 1800: continue
        last_notified[sid] = datetime.now()
        g = "B" if is_b else ("A" if is_a else "C")
        notify_groups.setdefault(g,[]).append(r)
    if notify_groups:
        lines = [f"📊 <b>買進訊號通知</b>",f"🕐 {datetime.now().strftime('%H:%M:%S')}  共{len(results)}支({elapsed}秒)",""]
        for grp,label,emoji in [("B","費波南+UT Bot雙重確認","🔴"),("A","UT+威廉+指標背離","🟠"),("C","UT+費波南+威廉三重確認","🟣")]:
            if grp not in notify_groups: continue
            lines.append(f"{emoji} <b>{label}</b>")
            for r in notify_groups[grp][:5]:
                s = (r.get("confirm") or {}).get("score",0)
                c = r.get("change_pct",0)
                lines.append(f"  ★ <b>{r['name']}({r['id']})</b> ${r.get('close',0):.1f} "
                             f"{'▲' if c>0 else '▼'}{abs(c):.2f}% 評分{s}分")
            lines.append("")
        lines.append("⚠ 僅供參考")
        send_telegram("\n".join(lines))

    # ── 條件 E：日週 MACD 共振（順勢），用剛掃完的 cache 跑，推 Telegram ──
    try:
        with cache_lock: cached_e = list(cache.values())
        e_hits, _ = scan_condition_e(cached_e)
        if e_hits:
            push_scan_summary("E", e_hits)
    except Exception as e:
        print(f"  ⚠ 自動 E 掃描失敗: {e}")

def scheduler_loop():
    global scheduler_running
    while scheduler_running:
        if is_market_hour(): auto_scan_job()
        else: print(f"[{datetime.now().strftime('%H:%M:%S')}] 非盤中，跳過")
        for _ in range(scheduler_interval * 60):
            if not scheduler_running: break
            time.sleep(1)


# ══════════════════════════════════════════════════════════
# ▌ Flask API 路由
# ══════════════════════════════════════════════════════════
@app.route("/api/scan", methods=["GET"])
def api_scan():
    for _ in range(12):
        if preload_done: break
        time.sleep(5); print(f"  ⏳ 等待預載... ({len(hist_cache)}/{len(watchlist)})")
    try:
        results = batch_scan_all()
        try:
            n_rec = record_divergence(results)
            if n_rec: print(f"  🗄 背離DB：寫入 {n_rec} 檔（背離≥{DIV_DB_MIN}）")
        except Exception: pass
        for r in results:
            try: add_to_watchlist2(r, min_score=3)
            except: pass
        display = [r for r in results if
                   r.get("ut_buy") or r.get("fib",{}).get("buy_signals") or
                   r.get("wr",{}).get("buy_signals") or (r.get("confirm") or {}).get("score",0)>=3]
        return jsonify(sanitize({"data":display,"scanned_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total":len(results),"signals":len([r for r in results if r.get("signals")]),
            "ut_combo":sum(1 for r in results if r.get("ut_buy")),
            "full_loaded":full_list_loaded,"preload_done":preload_done,
            "watchlist_size":len(watchlist)}))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error":str(e),"data":[],"total":0}), 500

@app.route("/api/stock/<sid>", methods=["GET"])
def api_single(sid):
    stock = next((s for s in watchlist if s["id"]==sid), {"id":sid,"name":sid,"ex":"tse"})
    try:
        return jsonify(sanitize(analyze_stock(stock)))
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    return jsonify(watchlist)

@app.route("/api/watchlist/add", methods=["POST"])
def api_add():
    data = request.json; sid = data.get("id","").strip()
    if not sid: return jsonify({"error":"缺少代號"}), 400
    if any(s["id"]==sid for s in watchlist): return jsonify({"error":"已在清單"}), 400
    watchlist.append({"id":sid,"name":data.get("name",sid),"ex":data.get("ex","tse")})
    return jsonify({"ok":True})

@app.route("/api/watchlist/remove/<sid>", methods=["DELETE"])
def api_remove(sid):
    global watchlist
    watchlist = [s for s in watchlist if s["id"]!=sid]
    with cache_lock: cache.pop(sid, None)
    return jsonify({"ok":True})

@app.route("/api/watchlist2", methods=["GET"])
def api_watchlist2_get():
    with watchlist2_lock: items = list(watchlist2)
    for item in items:
        cached = cache.get(item["id"],{})
        if cached:
            item.update({"ut_buy":cached.get("ut_buy",False),"ut_trail":cached.get("ut_trail",0),
                "close":cached.get("close",item.get("close",0)),
                "change_pct":cached.get("change_pct",0),
                "score":(cached.get("confirm") or {}).get("score",item.get("score",0))})
    items.sort(key=lambda x:(-int(x.get("ut_buy",False)),-x.get("score",0)))
    return jsonify({"data":items,"total":len(items),
        "ut_ready":sum(1 for x in items if x.get("ut_buy"))})

@app.route("/api/watchlist2/remove/<sid>", methods=["DELETE"])
def api_watchlist2_remove(sid):
    global watchlist2
    with watchlist2_lock:
        watchlist2 = [s for s in watchlist2 if s["id"]!=sid]
        save_watchlist2(watchlist2)
    return jsonify({"ok":True})

@app.route("/api/watchlist2/note", methods=["POST"])
def api_watchlist2_note():
    data = request.json; sid = data.get("id"); note = data.get("note","")
    with watchlist2_lock:
        item = next((s for s in watchlist2 if s["id"]==sid), None)
        if item: item["note"] = note; save_watchlist2(watchlist2)
    return jsonify({"ok":True})

@app.route("/api/watchlist2/clear", methods=["POST"])
def api_watchlist2_clear():
    global watchlist2
    with watchlist2_lock: watchlist2 = []; save_watchlist2(watchlist2)
    return jsonify({"ok":True})

@app.route("/api/scan/fibonacci", methods=["GET"])
def api_scan_fibonacci():
    with cache_lock: cached = list(cache.values())
    hits = sorted([r for r in cached if r.get("fib",{}).get("buy_signals")],
                  key=lambda x:(-len(x.get("fib",{}).get("buy_signals",[])),
                                -x.get("fib",{}).get("signals_count",0)))
    return jsonify(sanitize({"data":hits,"total":len(cached),"fib_count":len(hits),
                    "scanned_at":datetime.now().strftime("%H:%M:%S")}))

@app.route("/api/scan/williams", methods=["GET"])
def api_scan_williams():
    with cache_lock: cached = list(cache.values())
    hits = sorted([r for r in cached if r.get("wr",{}).get("buy_signals")],
                  key=lambda x:-len(x.get("wr",{}).get("buy_signals",[])))
    return jsonify(sanitize({"data":hits,"total":len(cached),"wr_count":len(hits),
                    "scanned_at":datetime.now().strftime("%H:%M:%S")}))

@app.route("/api/scan/utbot", methods=["GET"])
def api_scan_utbot():
    with cache_lock: cached = list(cache.values())
    combo = sorted([r for r in cached if r.get("ut_buy")],
                   key=lambda x:-(x.get("confirm",{}).get("score",0)))
    return jsonify(sanitize({"data":combo,"total":len(cached),"combo_count":len(combo),
                    "scanned_at":datetime.now().strftime("%H:%M:%S")}))

@app.route("/api/scan/buysignal", methods=["GET"])
def api_scan_buysignal():
    mode = request.args.get("mode","A")
    exclude_traditional = request.args.get("exclude_traditional","1") != "0"
    use_db = request.args.get("use_db","1") != "0"
    # 條件 A 擴充：過去 7 天背離≥5 的個股（DB），今天有 UT 買訊就補進來
    recent_div = get_recent_divergence() if (mode == "A" and use_db) else {}
    with cache_lock: cached = list(cache.values())
    results = []
    for r in cached:
        if r.get("volume",0) < 20: continue
        if exclude_traditional:
            ind = industry_map.get(r.get("id",""), "")
            if is_excluded_industry(ind): continue

        has_fib = bool(r.get("fib",{}).get("buy_signals"))
        has_ut  = bool(r.get("ut_buy"))
        has_wr  = bool(r.get("wr",{}).get("buy_signals"))
        div     = r.get("div",{})
        has_div = bool(div.get("buy_signals")) or div.get("score", 0) >= 2

        close      = r.get("close",  0)
        ma5        = r.get("ma5",    0)
        ma10       = r.get("ma10",   0)
        ma20       = r.get("ma20",   0)
        ut_trail   = r.get("ut_trail", 0)
        change_pct = r.get("change_pct", 0)

        fib_detail  = r.get("fib", {}).get("details", {})
        fib_ma3     = fib_detail.get("ma3", ma5)
        fib_ma5     = fib_detail.get("ma5", ma5)
        fib_above_ma = close > 0 and (
            (fib_ma3 > 0 and close >= fib_ma3 * 0.995) or
            (fib_ma5 > 0 and close >= fib_ma5 * 0.995) or
            (ma5     > 0 and close >= ma5     * 0.995)
        )

        ut_above_trail = close > 0 and (
            ut_trail <= 0 or close >= ut_trail * 0.995
        )

        not_falling = change_pct >= -1.0

        above_short_ma = close > 0 and (
            (ma5  > 0 and close >= ma5  * 0.995) or
            (ma10 > 0 and close >= ma10 * 0.995)
        )

        vol_ratio_ok = r.get("vol_ratio", 0) >= 1.2
        rising_strong = change_pct >= 1.0

        # MA 緊纏 + 沒突破才排除（橫盤無動能）。放行條件：
        #   (量比≥2.5 AND 漲幅≥2%)  ← 爆量起跑
        #   OR 前一日收盤 < 前一日MA5  OR 前一日收盤 < 前一日MA10  ← 昨天還在均線下方，今天剛突破
        ma60 = r.get("ma60", 0)
        prev_close = r.get("prev_close", 0)
        prev_ma5   = r.get("prev_ma5", 0)
        prev_ma10  = r.get("prev_ma10", 0)
        below_ma_yesterday = (
            (prev_ma5  > 0 and prev_close > 0 and prev_close < prev_ma5) or
            (prev_ma10 > 0 and prev_close > 0 and prev_close < prev_ma10)
        )
        ma_vals = [v for v in (ma5, ma10, ma20, ma60) if v > 0]
        ma_loose = True
        if len(ma_vals) >= 3:
            ma_max = max(ma_vals); ma_min = min(ma_vals)
            if ma_min > 0:
                cluster_pct = (ma_max - ma_min) / ma_min * 100
                ma_loose = (cluster_pct >= 2.5) or (
                    r.get("vol_ratio", 0) >= 2.5 and change_pct >= 2.0
                ) or below_ma_yesterday

        db_delayed = False
        if mode == "A":
            triggered = (has_ut and ut_above_trail and
                         has_wr and has_div and
                         not_falling and above_short_ma and
                         vol_ratio_ok and rising_strong and
                         ma_loose)
            # DB 擴充：過去7天背離≥5 + 今日UT買訊（不要求今日WR/今日背離）
            if not triggered and r.get("id") in recent_div and \
               has_ut and ut_above_trail and not_falling and above_short_ma:
                triggered = True
                db_delayed = True
        elif mode == "B":
            triggered = (has_ut and ut_above_trail and
                         has_fib and fib_above_ma and
                         not_falling and above_short_ma and
                         vol_ratio_ok and rising_strong and
                         ma_loose)
            # 必要條件：週線威廉波浪完成（只在前置閘門已過時才抓週線，省流量）
            if triggered:
                sid_b = r.get("id")
                ex_b  = next((s.get("ex","tse") for s in watchlist if s["id"]==sid_b), "tse")
                wkw = _get_weekly_williams_state(sid_b, ex_b, preset="B")
                if not wkw.get("wave_complete"):
                    triggered = False
                else:
                    r["weekly_wr"] = wkw
        elif mode == "C":
            confirm_score = (r.get("confirm") or {}).get("score", 0)
            triggered = (has_ut and ut_above_trail and
                         has_fib and fib_above_ma and
                         has_wr and
                         not_falling and above_short_ma and
                         confirm_score > 4 and
                         vol_ratio_ok and rising_strong and
                         ma_loose)
        elif mode == "D":
            div_score    = div.get("score", 0)
            regular_cnt  = div.get("regular_count", 0)
            triggered = (has_div and
                         (div_score >= 4 or regular_cnt >= 1) and
                         not_falling)
        else:
            triggered = (has_ut and ut_above_trail and
                         has_fib and fib_above_ma and
                         has_wr and not_falling)

        if triggered:
            r["buy_mode"]    = mode
            r["buy_reasons"] = []
            if has_fib:
                fib_ids = " ".join(s["id"] for s in r.get("fib",{}).get("buy_signals",[]))
                r["buy_reasons"].append(f"費波南[{fib_ids}]")
            if has_ut:
                r["buy_reasons"].append(f"UT Bot(追蹤線={ut_trail:.1f})")
            if has_wr:
                wr_ids = " ".join(s["id"] for s in r.get("wr",{}).get("buy_signals",[]))
                r["buy_reasons"].append(f"威廉[{wr_ids}]")
            if has_div:
                r["buy_reasons"].append(f"背離({div.get('score',0)}分)")
            if r.get("weekly_wr"):
                wkw = r["weekly_wr"]
                r["buy_reasons"].append(
                    f"週線威廉波浪完成(深底{wkw.get('deep_low_val',0):.0f}/"
                    f"{wkw.get('peak_count',0)}峰)")
            if db_delayed:
                info = recent_div.get(r.get("id"), {})
                r["db_delayed"] = True
                r["buy_reasons"].append(f"DB背離{info.get('div','')}分@{info.get('last_date','')}+今日UT")
            results.append(r)

    if mode == "D":
        results.sort(key=lambda x: (
            -(x.get("div") or {}).get("score", 0),
            -(x.get("div") or {}).get("regular_count", 0),
            -x.get("change_pct", 0),
        ))
    else:
        results.sort(key=lambda x: (
            -(x.get("confirm") or {}).get("score", 0),
            -x.get("change_pct", 0),
        ))
    push_scan_summary(mode, results)
    return jsonify(sanitize({"data":results,"total":len(cached),"hit_count":len(results),"mode":mode,
                    "scanned_at":datetime.now().strftime("%H:%M:%S")}))

@app.route("/api/scan/divergence", methods=["GET"])
def api_scan_divergence():
    div_type  = request.args.get("type","all")
    min_score = int(request.args.get("min_score",2))
    with cache_lock: cached = list(cache.values())
    if not cached:
        return jsonify({"error":"請先執行主掃描","data":[],"hit_count":0}), 400

    def calc_div_for(r):
        sid = r.get("id"); ex = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        try:
            hist = fetch_history(sid, ex=ex)
            if len(hist) < 30: return r, {}, {}
            closes  = hist["Close"]; volumes = hist["Volume"]
            highs   = hist["High"]  if "High" in hist.columns else closes
            lows_s  = hist["Low"]   if "Low"  in hist.columns else closes
            ind = calc_all_indicators(closes, highs, lows_s, volumes)
            div = calc_divergence_signals(ind, closes, highs, lows_s, volumes,
                                           lookback=120, max_bars=15)
            n = len(closes)
            extra = {}
            if n >= 14:
                c_v = closes.values
                ma5_y  = float(c_v[max(0,n-6):n-1].mean())
                ma10_y = float(c_v[max(0,n-11):n-1].mean())
                c_y    = float(c_v[-2])
                c_now  = float(c_v[-1])
                opens  = hist["Open"] if "Open" in hist.columns else closes
                o_v    = opens.values
                o_8    = float(o_v[-9])  if n >= 9  else None
                o_13   = float(o_v[-14]) if n >= 14 else None
                extra  = {"cond_ma": (c_y<=ma5_y) or (c_y<=ma10_y),
                          "cond_fib_break": bool(o_8 and c_now>=o_8 and o_13 and c_now>=o_13)}
            return r, div, extra
        except: return r, {}, {}

    hits_raw = []
    futures  = {_global_executor.submit(calc_div_for, r): r for r in cached}
    for future in as_completed(futures):
        try:
            r, div, extra = future.result()
            score   = div.get("score",0)
            signals = list(div.get("buy_signals",[]))
            if score < min_score or not signals: continue
            cond_ma  = extra.get("cond_ma",False)
            cond_str = (score>=6) or (div.get("regular_count",0)>=3) or extra.get("cond_fib_break",False)
            if not (cond_ma and cond_str): continue
            if div_type=="regular":
                signals = [s for s in signals if s["type"]=="regular"]
            elif div_type=="hidden":
                signals = [s for s in signals if s["type"]=="hidden"]
            if not signals: continue

            sid = r.get("id")
            if sid:
                with cache_lock:
                    if sid in cache:
                        cache[sid]["div"] = div

            hits_raw.append({**r,"div":div,"div_score":score,
                "div_summary":div.get("summary",""),"div_signals":signals,
                "regular_count":div.get("regular_count",0),
                "hidden_count":div.get("hidden_count",0),"filter_detail":extra})
        except: pass

    hits_raw.sort(key=lambda x:(-x.get("div_score",0),-(x.get("confirm") or {}).get("score",0)))
    return jsonify(sanitize({"data":hits_raw,"total":len(cached),"hit_count":len(hits_raw),
        "regular_count":sum(1 for h in hits_raw if h.get("regular_count",0)>0),
        "hidden_count":sum(1 for h in hits_raw if h.get("hidden_count",0)>0),
        "div_type":div_type,"min_score":min_score,
        "scanned_at":datetime.now().strftime("%H:%M:%S")}))

@app.route("/api/divergence_db", methods=["GET"])
def api_divergence_db():
    """查看背離資料庫累積內容（過去 7 天）"""
    try:
        with _div_db_lock, sqlite3.connect(DIVERGENCE_DB) as con:
            cur = con.execute("""SELECT sid,name,date,daily_div,weekly_div,close
                FROM divergence_history ORDER BY date DESC, daily_div DESC""")
            rows = [{"sid":x[0],"name":x[1],"date":x[2],"daily_div":x[3],
                     "weekly_div":x[4],"close":x[5]} for x in cur.fetchall()]
        uniq = len({r["sid"] for r in rows})
        return jsonify({"total_rows":len(rows),"unique_stocks":uniq,
                        "min_div":DIV_DB_MIN,"days":DIV_DB_DAYS,"data":rows})
    except Exception as e:
        return jsonify({"error":str(e),"data":[]}), 500

@app.route("/api/divergence_db/clear", methods=["POST","GET"])
def api_divergence_db_clear():
    try:
        with _div_db_lock, sqlite3.connect(DIVERGENCE_DB) as con:
            con.execute("DELETE FROM divergence_history")
        return jsonify({"ok":True,"msg":"背離DB已清空"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


def _dw_divergence(sid: str, ex: str = "tse"):
    """條件 D 核心（嚴格）：日K背離分數≥5 且 週K背離分數≥2。回傳 (div_d, div_w) 或 None。
    供條件 D 路由與條件 F 的聯集共用，結果一致。"""
    try:
        from divergence_screener import calc_ind as ds_calc_ind, calc_div_score as ds_calc_div_score
    except Exception:
        return None
    try:
        df = fetch_history(sid, period="1y", ex=ex)
        if len(df) < 60: return None
        closes  = df["Close"]; volumes = df["Volume"]
        highs   = df["High"] if "High" in df.columns else closes
        lows_s  = df["Low"]  if "Low"  in df.columns else closes
        ind_d   = ds_calc_ind(closes, highs, lows_s, volumes)
        div_d   = ds_calc_div_score(ind_d, closes, highs, lows_s, volumes, max_bars=15)
        if div_d.get("score", 0) < 5: return None
        df.index = pd.to_datetime(df.index)
        wdf = df.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min",
                                          "Close":"last","Volume":"sum"}).dropna()
        if len(wdf) < 12: return None
        wc = wdf["Close"]; wv = wdf["Volume"]
        wh = wdf["High"] if "High" in wdf.columns else wc
        wl = wdf["Low"]  if "Low"  in wdf.columns else wc
        ind_w = ds_calc_ind(wc, wh, wl, wv)
        div_w = ds_calc_div_score(ind_w, wc, wh, wl, wv, max_bars=10)
        if div_w.get("score", 0) < 2: return None
        return div_d, div_w
    except Exception:
        return None


@app.route("/api/scan/condition_d", methods=["GET"])
def api_scan_condition_d():
    """條件 D：日K背離分數≥5 + 週K背離分數≥2（三指標+日週雙重 / 日週強烈共振）
    使用 divergence_screener 的相同函式以保持結果一致"""
    try:
        from divergence_screener import calc_ind as ds_calc_ind, calc_div_score as ds_calc_div_score
    except Exception as e:
        return jsonify({"error":f"divergence_screener 匯入失敗：{e}","data":[],"hit_count":0,"mode":"D"}), 500

    exclude_traditional = request.args.get("exclude_traditional","1") != "0"
    with cache_lock: cached = list(cache.values())
    if not cached:
        return jsonify({"error":"請先執行主掃描","data":[],"hit_count":0,"mode":"D"}), 400

    def _ind_ok(sid):
        if not exclude_traditional: return True
        return not is_excluded_industry(industry_map.get(sid, ""))

    candidates = [r for r in cached
                  if (r.get("div") or {}).get("score", 0) >= 2
                  and r.get("volume", 0) >= 20
                  and _ind_ok(r.get("id",""))]
    if not candidates:
        return jsonify(sanitize({"data":[],"total":len(cached),"hit_count":0,"mode":"D",
                                  "scanned_at":datetime.now().strftime("%H:%M:%S")}))

    def analyze_dw(r):
        sid = r.get("id")
        ex = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        res = _dw_divergence(sid, ex)
        if not res: return None
        div_d, div_w = res
        return r, div_d, div_w

    hits = []
    futures = {_global_executor.submit(analyze_dw, r): r for r in candidates}
    for future in as_completed(futures):
        try:
            res = future.result(timeout=30)
            if not res: continue
            r, div_d, div_w = res
            ws = div_w.get("score", 0)
            ds = div_d.get("score", 0)
            # 統一欄位名：calc_div_score 用 "signals"，cache 用 "buy_signals"
            div_d_norm = {**div_d, "buy_signals": div_d.get("signals", [])}
            r_out = {**r, "div": div_d_norm}
            r_out["weekly_score"]    = ws
            r_out["weekly_signals"]  = div_w.get("signals", [])
            r_out["weekly_summary"]  = div_w.get("summary", "")
            r_out["strong_resonance"]= ws >= 4
            r_out["double_resonance"]= True
            r_out["buy_mode"]        = "D"
            tag = "日週強烈共振" if ws >= 4 else "日週雙重共振"
            r_out["buy_reasons"]     = [f"日背離({ds}分)", f"週背離({ws}分)", tag]
            hits.append(r_out)
        except Exception:
            pass

    # ── DB 擴充：過去7天背離≥5 + 今日UT買訊（不要求今日週背離）──
    use_db = request.args.get("use_db","1") != "0"
    db_added = 0
    if use_db:
        recent_div = get_recent_divergence()
        hit_ids = {h.get("id") for h in hits}
        for r in cached:
            sid = r.get("id")
            if sid in hit_ids: continue
            if sid not in recent_div: continue
            if not r.get("ut_buy"): continue
            if r.get("volume",0) < 20: continue
            if not _ind_ok(sid): continue
            info = recent_div[sid]
            r_out = {**r}
            r_out["weekly_score"]     = 0
            r_out["strong_resonance"] = False
            r_out["double_resonance"] = False
            r_out["db_delayed"]       = True
            r_out["buy_mode"]         = "D"
            r_out["buy_reasons"]      = [f"DB背離{info['div']}分@{info['last_date']}", "今日UT確認"]
            hits.append(r_out); db_added += 1

    hits.sort(key=lambda x: (
        -int(x.get("strong_resonance", False)),
        -x.get("weekly_score", 0),
        -(x.get("div") or {}).get("score", 0),
        -x.get("change_pct", 0),
    ))
    push_scan_summary("D", hits)
    return jsonify(sanitize({"data":hits,"total":len(cached),"hit_count":len(hits),
                              "mode":"D","candidates":len(candidates),"db_added":db_added,
                              "strong_count":sum(1 for h in hits if h.get("strong_resonance")),
                              "scanned_at":datetime.now().strftime("%H:%M:%S")}))


# ── 週 MACD 多頭快取（7 天內共用，因週線一週才變一次）─────
_weekly_macd_cache = {}   # sid -> (bool, timestamp)
_weekly_macd_lock  = threading.Lock()
_WEEKLY_CACHE_TTL  = 7 * 24 * 3600  # 7 天

# ── 週線威廉波浪快取（同樣 7 天，週線一週才變一次）─────
_weekly_wr_cache = {}     # sid -> (dict, timestamp)
_weekly_wr_lock  = threading.Lock()

def _check_weekly_macd_bull(sid: str, ex: str = "tse") -> bool:
    """回傳該股票週 MACD 是否多頭（DIF > DEA）。7 天內共用快取。"""
    return _get_weekly_macd_state(sid, ex).get("bull", False)

def _get_weekly_macd_state(sid: str, ex: str = "tse") -> dict:
    """回傳週 MACD 完整狀態：bull/golden_cross/above_zero/hist_rising/dif/dea/hist。
    7 天內共用快取（週線一週才變一次）。"""
    now_ts = time.time()
    with _weekly_macd_lock:
        entry = _weekly_macd_cache.get(sid)
        if entry and (now_ts - entry[1]) < _WEEKLY_CACHE_TTL and isinstance(entry[0], dict):
            return entry[0]
    empty = {"bull":False,"golden_cross":False,"above_zero":False,"hist_rising":False,
             "dif":0.0,"dea":0.0,"hist":0.0}
    try:
        df = fetch_history(sid, period="1y", ex=ex)
        if len(df) < 60:
            with _weekly_macd_lock: _weekly_macd_cache[sid] = (empty, now_ts)
            return empty
        df.index = pd.to_datetime(df.index)
        wdf = df.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min",
                                          "Close":"last","Volume":"sum"}).dropna()
        if len(wdf) < 30:
            with _weekly_macd_lock: _weekly_macd_cache[sid] = (empty, now_ts)
            return empty
        wc = wdf["Close"].values.astype(float)
        ema12 = pd.Series(wc).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(wc).ewm(span=26, adjust=False).mean().values
        dif   = ema12 - ema26
        dea   = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        hist  = (dif - dea) * 2
        state = {
            "bull":         bool(dif[-1] > dea[-1]),
            "golden_cross": bool(dif[-2] <= dea[-2] and dif[-1] > dea[-1]),
            "above_zero":   bool(dif[-1] > 0 and dea[-1] > 0),
            "hist_rising":  bool(hist[-1] > hist[-2]),
            "dif":  round(float(dif[-1]), 4),
            "dea":  round(float(dea[-1]), 4),
            "hist": round(float(hist[-1]), 4),
        }
        with _weekly_macd_lock: _weekly_macd_cache[sid] = (state, now_ts)
        return state
    except Exception:
        with _weekly_macd_lock: _weekly_macd_cache[sid] = (empty, now_ts)
        return empty


# 週線威廉波浪參數預設（B / F 各自獨立，互不影響）：
#   B：寬鬆（沿用日線參數），B 的嚴格度來自 UT+費波南，週威廉只當多一道確認
#   F：收緊（剛脫離 + 波峰≥4 + 深底窗口6-26週），避免普漲日假脫離
_WEEKLY_WR_PRESETS = {
    "B": {"window_min":8, "window_max":21, "min_peaks":3,
          "deep_low_thr":-85.0, "escape_thr":-50.0, "fresh_escape":False},
    "F": {"window_min":6, "window_max":26, "min_peaks":4,
          "deep_low_thr":-85.0, "escape_thr":-50.0, "fresh_escape":True},
}

def _get_weekly_williams_state(sid: str, ex: str = "tse", preset: str = "F") -> dict:
    """回傳週線威廉波浪狀態：wave_complete/deep_low_val/bars_from_low/peak_count/wr_now。
    把日線 history resample 成週 K，丟給 calc_williams_signals_v4。
    preset 決定波浪參數（B 寬 / F 嚴），cache 以 (sid, preset) 為 key 避免互相污染。"""
    now_ts = time.time()
    ckey = (sid, preset)
    with _weekly_wr_lock:
        entry = _weekly_wr_cache.get(ckey)
        if entry and (now_ts - entry[1]) < _WEEKLY_CACHE_TTL and isinstance(entry[0], dict):
            return entry[0]
    empty = {"wave_complete":False,"deep_low_val":0.0,"bars_from_low":0,
             "peak_count":0,"wr_now":-50.0}
    try:
        df = fetch_history(sid, period="2y", ex=ex)   # 週線需 ~60 根 → 抓 2 年
        if len(df) < 60:
            with _weekly_wr_lock: _weekly_wr_cache[ckey] = (empty, now_ts)
            return empty
        df.index = pd.to_datetime(df.index)
        wdf = df.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min",
                                          "Close":"last","Volume":"sum"}).dropna()
        if len(wdf) < 30:
            with _weekly_wr_lock: _weekly_wr_cache[ckey] = (empty, now_ts)
            return empty
        wave_params = _WEEKLY_WR_PRESETS.get(preset, _WEEKLY_WR_PRESETS["F"])
        wr_res = calc_williams_signals_v4(wdf["Close"], wdf["High"], wdf["Low"],
                                           wave_params=wave_params)
        wi = wr_res.get("wave_info", {})
        state = {
            "wave_complete": bool(wi.get("wave_complete", False)),
            "deep_low_val":  wi.get("deep_low_val", 0.0),
            "bars_from_low": wi.get("bars_from_low", 0),
            "peak_count":    wi.get("peak_count", 0),
            "wr_now":        wr_res.get("wr_now", -50.0),
        }
        with _weekly_wr_lock: _weekly_wr_cache[ckey] = (state, now_ts)
        return state
    except Exception:
        with _weekly_wr_lock: _weekly_wr_cache[ckey] = (empty, now_ts)
        return empty


@app.route("/api/scan/condition_c_strict", methods=["GET"])
def api_scan_condition_c_strict():
    """條件 C 嚴格版：在 cache 過濾通過後，再要求週 MACD 多頭。
    第一次跑會慢（每支抓 1y 資料），同日內第二次以後用 cache。"""
    with cache_lock: cached = list(cache.values())
    if not cached:
        return jsonify({"error":"請先執行主掃描","data":[],"hit_count":0,"mode":"C+"}), 400

    # 先用 cache 跑一次條件 C 的判斷
    candidates = []
    for r in cached:
        if r.get("volume",0) < 20: continue
        has_fib = bool(r.get("fib",{}).get("buy_signals"))
        has_ut  = bool(r.get("ut_buy"))
        has_wr  = bool(r.get("wr",{}).get("buy_signals"))
        close      = r.get("close", 0)
        ma5        = r.get("ma5",   0); ma10 = r.get("ma10",0)
        ma20       = r.get("ma20",  0); ma60 = r.get("ma60",0)
        ut_trail   = r.get("ut_trail", 0)
        change_pct = r.get("change_pct", 0)
        fib_detail = r.get("fib",{}).get("details",{})
        fib_ma3    = fib_detail.get("ma3", ma5); fib_ma5 = fib_detail.get("ma5", ma5)
        fib_above_ma = close > 0 and (
            (fib_ma3 > 0 and close >= fib_ma3 * 0.995) or
            (fib_ma5 > 0 and close >= fib_ma5 * 0.995) or
            (ma5     > 0 and close >= ma5     * 0.995))
        ut_above_trail = close > 0 and (ut_trail <= 0 or close >= ut_trail * 0.995)
        not_falling    = change_pct >= -1.0
        above_short_ma = close > 0 and (
            (ma5  > 0 and close >= ma5  * 0.995) or
            (ma10 > 0 and close >= ma10 * 0.995))
        vol_ratio_ok  = r.get("vol_ratio",0) >= 1.2
        rising_strong = change_pct >= 1.0
        ma_vals = [v for v in (ma5, ma10, ma20, ma60) if v > 0]
        ma_loose = True
        if len(ma_vals) >= 3 and min(ma_vals) > 0:
            ma_loose = (max(ma_vals) - min(ma_vals)) / min(ma_vals) * 100 >= 2.5
        confirm_score = (r.get("confirm") or {}).get("score",0)
        if (has_ut and ut_above_trail and has_fib and fib_above_ma and has_wr and
            not_falling and above_short_ma and confirm_score > 4 and
            vol_ratio_ok and rising_strong and ma_loose):
            candidates.append(r)

    if not candidates:
        return jsonify(sanitize({"data":[],"total":len(cached),"hit_count":0,"mode":"C+",
                                  "scanned_at":datetime.now().strftime("%H:%M:%S")}))

    # 對 candidates on-demand 檢查週 MACD 多頭
    def check(r):
        sid = r.get("id"); ex = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        return r if _check_weekly_macd_bull(sid, ex) else None

    hits = []
    futures = {_global_executor.submit(check, r): r for r in candidates}
    for f in as_completed(futures):
        try:
            res = f.result(timeout=20)
            if res:
                res = dict(res); res["weekly_macd_bull"] = True; res["buy_mode"] = "C+"
                hits.append(res)
        except Exception: pass

    hits.sort(key=lambda x: (
        -(x.get("confirm") or {}).get("score", 0),
        -x.get("change_pct", 0),
    ))
    push_scan_summary("C+", hits)
    return jsonify(sanitize({"data":hits,"total":len(cached),"hit_count":len(hits),
                              "mode":"C+","candidates":len(candidates),
                              "scanned_at":datetime.now().strftime("%H:%M:%S")}))


def _e_priority(r):
    """進場優先分(0-100)：把看線型會檢查的東西量化，解決一次20檔無法逐一看圖的問題。
    權重偏向「不追高」：進場位階40 / 訊號強度20 / 趨勢結構20 / 多重確認+量能20。
    回傳 (優先分, 位階標籤, 離MA20乖離%)。"""
    e_score = r.get("e_score", 0)
    close = r.get("close",0); ma5=r.get("ma5",0); ma10=r.get("ma10",0)
    ma20=r.get("ma20",0); ma60=r.get("ma60",0)
    # 1. 訊號強度 (0-20)：日週共振+背離越強越高
    p_sig = e_score / 9.0 * 20
    # 2. 趨勢結構 (0-20)：均線多頭排列越完整越好
    if   ma5>ma10>ma20>ma60>0: p_struct = 20
    elif close>ma20>ma60>0:    p_struct = 12
    elif close>ma20>0:         p_struct = 6
    else:                      p_struct = 0
    # 3. 進場位階 (0-40, 主導)：離 MA20 乖離率，越貼近月線越好，追高大幅扣分
    bias = ((close-ma20)/ma20*100) if ma20>0 else 99
    if   bias <= 4:  p_level, level_tag = 40, "剛突破✓"
    elif bias <= 7:  p_level, level_tag = 32, "小幅乖離"
    elif bias <= 10: p_level, level_tag = 22, "乖離中等"
    elif bias <= 15: p_level, level_tag = 10, "乖離偏大"
    else:            p_level, level_tag = 2,  "追高⚠"
    # 4. 多重確認 + 量能 (0-20)
    p_conf = 0
    if r.get("ut_buy"):                          p_conf += 6
    if (r.get("fib") or {}).get("buy_signals"):  p_conf += 4
    if (r.get("wr")  or {}).get("buy_signals"):  p_conf += 4
    vr = r.get("vol_ratio", 0)
    p_conf += 6 if vr >= 2 else (3 if vr >= 1.5 else 0)
    p_conf = min(p_conf, 20)
    prio = round(p_sig + p_struct + p_level + p_conf)
    return prio, level_tag, round(bias, 1)


def scan_condition_e(cached, exclude_traditional=True, min_vol_ratio=1.5,
                      require_above_zero=False, min_score=6, max_bias=12.0):
    """條件 E 掃描核心，回傳 (hits, candidate_count)。供 API route 與自動排程共用。"""
    def _ind_ok(sid):
        if not exclude_traditional: return True
        return not is_excluded_industry(industry_map.get(sid, ""))

    # 先用 cache 篩日線金叉 + 站上MA20 + 量比（快，免抓週線）
    candidates = []
    for r in cached:
        if r.get("volume", 0) < 20: continue
        if r.get("vol_ratio", 0) < min_vol_ratio: continue
        if not _ind_ok(r.get("id","")): continue
        close = r.get("close", 0); ma20 = r.get("ma20", 0)
        dif = r.get("dif", 0); dea = r.get("dea", 0)
        prev_dif = r.get("prev_dif", 0); prev_dea = r.get("prev_dea", 0)
        # 日線 MACD 金叉：昨日 DIF≤DEA、今日 DIF>DEA
        daily_golden = (prev_dif <= prev_dea) and (dif > dea)
        # 現價站上 MA20（避免假訊號）
        above_ma20 = close > 0 and ma20 > 0 and close >= ma20
        if daily_golden and above_ma20:
            candidates.append(r)

    if not candidates:
        return [], 0

    # 對候選 on-demand 檢查週 MACD：須多頭(可選零軸上)
    def check(r):
        sid = r.get("id")
        ex = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        w = _get_weekly_macd_state(sid, ex)
        if not w.get("bull"): return None
        if require_above_zero and not w.get("above_zero"): return None
        r_out = dict(r)
        dif = r.get("dif",0); dea = r.get("dea",0)
        div_score = (r.get("div") or {}).get("score", 0)
        # 品質加分
        flags = []
        score = 0
        if w.get("above_zero"):   score += 2; flags.append("週零軸上(強多)")
        if w.get("golden_cross"): score += 2; flags.append("週剛金叉")
        if w.get("hist_rising"):  score += 1; flags.append("週紅柱放大")
        if dif < 0:               score += 1; flags.append("日金叉在零軸下(起漲)")
        if div_score >= 2:        score += 2; flags.append(f"日底背離{div_score}")
        if r.get("vol_ratio",0) >= 1.0 and r.get("change_pct",0) > 0:
            score += 1; flags.append("帶量紅K")
        if score < min_score: return None
        r_out["weekly_macd"]   = w
        r_out["e_score"]       = score
        prio, level_tag, bias = _e_priority(r_out)
        if bias > max_bias: return None   # 乖離過大(追高)直接濾掉,不只是排後面
        r_out["prio"]          = prio
        r_out["level_tag"]     = level_tag
        r_out["bias20"]        = bias
        r_out["buy_mode"]      = "E"
        r_out["buy_reasons"]   = ["日週MACD共振", "週MACD多頭", "日MACD金叉", "站上MA20"] + flags
        return r_out

    hits = []
    futures = {_global_executor.submit(check, r): r for r in candidates}
    for f in as_completed(futures):
        try:
            res = f.result(timeout=20)
            if res: hits.append(res)
        except Exception: pass

    hits.sort(key=lambda x: (-x.get("prio",0), -x.get("e_score",0), -x.get("change_pct",0)))
    return hits, len(candidates)


@app.route("/api/scan/condition_e", methods=["GET"])
def api_scan_condition_e():
    """條件 E：日週 MACD 共振（順勢）
    必要：週DIF>DEA(多頭) + 日MACD金叉 + 站上MA20 + 量比≥1.5 + 離MA20乖離≤max_bias%
    加分：週零軸上(強多)、週剛金叉、週紅柱放大、日金叉在零軸下(起漲)、日底背離、帶量紅K
    最後只回 E評分≥min_score。on-demand 抓週線（同 D），第一次慢、之後 7 天 cache。"""
    exclude_traditional = request.args.get("exclude_traditional","1") != "0"
    min_vol_ratio = float(request.args.get("min_vol_ratio", "1.5"))
    require_above_zero = request.args.get("require_above_zero", "0") != "0"
    min_score = int(request.args.get("min_score", "6"))
    max_bias = float(request.args.get("max_bias", "12"))  # 離MA20乖離上限,超過視為追高直接濾掉
    with cache_lock: cached = list(cache.values())
    if not cached:
        return jsonify({"error":"請先執行主掃描","data":[],"hit_count":0,"mode":"E"}), 400

    hits, n_cand = scan_condition_e(cached, exclude_traditional, min_vol_ratio,
                                     require_above_zero, min_score, max_bias)
    push_scan_summary("E", hits)
    return jsonify(sanitize({"data":hits,"total":len(cached),"hit_count":len(hits),
                              "mode":"E","candidates":n_cand,
                              "scanned_at":datetime.now().strftime("%H:%M:%S")}))


def _f_pullback_ok(r):
    """昨日回檔確認：1日前收盤<近3日最高 且 (1日前收盤≤1日前MA5 或 1日前收盤≤1日前MA10)。
    近3日最高=2/3/4日前三日最高。意義：今天觸發買訊前，昨天才剛從近期高點/短均下方回檔 → 不追高。"""
    pc   = r.get("prev_close", 0)
    ph   = r.get("prev3_high", 0)
    pm5  = r.get("prev_ma5", 0)
    pm10 = r.get("prev_ma10", 0)
    if pc <= 0 or ph <= 0:
        return False
    return pc < ph and ((pm5 > 0 and pc <= pm5) or (pm10 > 0 and pc <= pm10))


def _f_union_abcde(r, e_ids, d_ids):
    """是否符合 A/B/C/D/E 任一。A/B/C 用 cache 即時判斷；D 用嚴格週背離命中集合；E 用 MACD共振命中集合。
    回傳符合的條件標籤 list。"""
    has_fib = bool(r.get("fib",{}).get("buy_signals"))
    has_ut  = bool(r.get("ut_buy"))
    has_wr  = bool(r.get("wr",{}).get("buy_signals"))
    div     = r.get("div",{})
    has_div = bool(div.get("buy_signals")) or div.get("score",0) >= 2
    close = r.get("close",0); ma5=r.get("ma5",0); ma10=r.get("ma10",0)
    ma20=r.get("ma20",0); ma60=r.get("ma60",0)
    ut_trail=r.get("ut_trail",0); change_pct=r.get("change_pct",0)
    confirm_score=(r.get("confirm") or {}).get("score",0)
    fib_d=r.get("fib",{}).get("details",{})
    fib_ma3=fib_d.get("ma3",ma5); fib_ma5=fib_d.get("ma5",ma5)
    fib_above_ma = close>0 and ((fib_ma3>0 and close>=fib_ma3*0.995) or
                                (fib_ma5>0 and close>=fib_ma5*0.995) or
                                (ma5>0 and close>=ma5*0.995))
    ut_above_trail = close>0 and (ut_trail<=0 or close>=ut_trail*0.995)
    not_falling = change_pct >= -1.0
    above_short_ma = close>0 and ((ma5>0 and close>=ma5*0.995) or (ma10>0 and close>=ma10*0.995))
    vol_ratio_ok = r.get("vol_ratio",0) >= 1.2
    rising_strong = change_pct >= 1.0
    prev_close=r.get("prev_close",0); prev_ma5=r.get("prev_ma5",0); prev_ma10=r.get("prev_ma10",0)
    below_ma_yesterday = ((prev_ma5>0 and prev_close>0 and prev_close<prev_ma5) or
                          (prev_ma10>0 and prev_close>0 and prev_close<prev_ma10))
    ma_vals=[v for v in (ma5,ma10,ma20,ma60) if v>0]
    ma_loose=True
    if len(ma_vals)>=3 and min(ma_vals)>0:
        cluster=(max(ma_vals)-min(ma_vals))/min(ma_vals)*100
        ma_loose=(cluster>=2.5) or (r.get("vol_ratio",0)>=2.5 and change_pct>=2.0) or below_ma_yesterday
    common = ut_above_trail and not_falling and above_short_ma and vol_ratio_ok and rising_strong and ma_loose
    is_a = has_ut and has_wr and has_div and common
    is_b = has_ut and has_fib and fib_above_ma and common
    is_c = has_ut and has_fib and has_wr and (confirm_score>4) and common
    is_d = r.get("id") in d_ids   # 嚴格D：日K背離≥5 + 週K背離≥2
    is_e = r.get("id") in e_ids
    tags = []
    if is_a: tags.append("A")
    if is_b: tags.append("B")
    if is_c: tags.append("C")
    if is_d: tags.append("D")
    if is_e: tags.append("E")
    return tags


@app.route("/api/scan/condition_f", methods=["GET"])
def api_scan_condition_f():
    """條件 F（嚴格）：週線威廉波浪完成 + 符合A/B/C/D/E任一 + 昨日回檔確認。
    昨日回檔 = 1日前收盤<近3日最高(2/3/4日前) 且 (1日前收盤≤1日前MA5 或 ≤1日前MA10)。
    D 為嚴格版(日K背離≥5+週K背離≥2)。流程：先昨日回檔過濾 → 算D/E命中集合 → 聯集 → 週線威廉(F參數,嚴)。"""
    exclude_traditional = request.args.get("exclude_traditional","1") != "0"
    with cache_lock: cached = list(cache.values())
    if not cached:
        return jsonify({"error":"請先執行主掃描","data":[],"hit_count":0,"mode":"F"}), 400

    def _ind_ok(sid):
        if not exclude_traditional: return True
        return not is_excluded_industry(industry_map.get(sid, ""))

    # 先用最便宜的條件過濾：流動性 + 昨日回檔
    pullback_pool = [r for r in cached
                     if r.get("volume",0) >= 20 and _ind_ok(r.get("id",""))
                     and _f_pullback_ok(r)]
    if not pullback_pool:
        return jsonify(sanitize({"data":[],"total":len(cached),"hit_count":0,"mode":"F",
                                  "candidates":0,"scanned_at":datetime.now().strftime("%H:%M:%S")}))

    # E 命中集合（E 內含週MACD，7天cache）
    try:
        e_hits, _ = scan_condition_e(cached, exclude_traditional=exclude_traditional)
        e_ids = {h.get("id") for h in e_hits}
    except Exception:
        e_ids = set()

    # 嚴格D命中集合（日K背離≥5+週K背離≥2）：只對 pullback_pool 中日背離≥2 的股抓週線，省流量
    d_cand = [r for r in pullback_pool if (r.get("div") or {}).get("score",0) >= 2]
    d_ids = set()
    if d_cand:
        dfuts = {_global_executor.submit(
                    _dw_divergence, r.get("id"),
                    next((s.get("ex","tse") for s in watchlist if s["id"]==r.get("id")), "tse")
                 ): r.get("id") for r in d_cand}
        for ft in as_completed(dfuts):
            try:
                if ft.result(timeout=30): d_ids.add(dfuts[ft])
            except Exception: pass

    # 聯集過濾：符合 A/B/C/D/E 任一
    candidates = []
    f_tags_by_id = {}
    for r in pullback_pool:
        tags = _f_union_abcde(r, e_ids, d_ids)
        if not tags: continue
        f_tags_by_id[r.get("id")] = tags
        candidates.append(r)
    if not candidates:
        return jsonify(sanitize({"data":[],"total":len(cached),"hit_count":0,"mode":"F",
                                  "candidates":0,"scanned_at":datetime.now().strftime("%H:%M:%S")}))

    def check(r):
        sid = r.get("id")
        ex  = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        wkw = _get_weekly_williams_state(sid, ex, preset="F")
        if not wkw.get("wave_complete"): return None
        r_out = dict(r)
        r_out["weekly_wr"]   = wkw
        r_out["buy_mode"]    = "F"
        r_out["f_tags"]      = f_tags_by_id.get(sid, [])
        r_out["buy_reasons"] = [
            f"週線威廉波浪完成(深底WR{wkw.get('deep_low_val',0):.0f}，"
            f"{wkw.get('bars_from_low',0)}週前，{wkw.get('peak_count',0)}個波峰，"
            f"今週WR{wkw.get('wr_now',0):.0f}脫離超賣)",
            f"符合條件 {'/'.join(f_tags_by_id.get(sid, []))}",
            "昨日回檔確認(昨收<前日高且≤短均)"]
        return r_out

    hits = []
    futures = {_global_executor.submit(check, r): r for r in candidates}
    for f in as_completed(futures):
        try:
            res = f.result(timeout=20)
            if res: hits.append(res)
        except Exception: pass

    hits.sort(key=lambda x: -x.get("change_pct", 0))
    push_scan_summary("F", hits)
    return jsonify(sanitize({"data":hits,"total":len(cached),"hit_count":len(hits),
                              "mode":"F","candidates":len(candidates),
                              "scanned_at":datetime.now().strftime("%H:%M:%S")}))


# ══════════════════════════════════════════════════════════
# ▌ 條件 G：(UT或費波南)∧(日威廉/背離/MACD共振/週威廉) + 昨日回檔 + 變盤訊號<4
# ══════════════════════════════════════════════════════════
def _g_signal_states(r):
    """條件 G 第1點所需的各訊號狀態（沿用 A~F 既有門檻）。
    common = A/B/C 共用的穩健門檻；ut/fib 為群1；wr/div 為群2 cache 可判定部分。"""
    has_fib = bool(r.get("fib",{}).get("buy_signals"))
    has_ut  = bool(r.get("ut_buy"))
    has_wr  = bool(r.get("wr",{}).get("buy_signals"))
    div     = r.get("div",{})
    has_div = bool(div.get("buy_signals")) or div.get("score",0) >= 2
    close=r.get("close",0); ma5=r.get("ma5",0); ma10=r.get("ma10",0)
    ma20=r.get("ma20",0); ma60=r.get("ma60",0)
    ut_trail=r.get("ut_trail",0); change_pct=r.get("change_pct",0)
    fib_d=r.get("fib",{}).get("details",{})
    fib_ma3=fib_d.get("ma3",ma5); fib_ma5=fib_d.get("ma5",ma5)
    fib_above_ma = close>0 and ((fib_ma3>0 and close>=fib_ma3*0.995) or
                                (fib_ma5>0 and close>=fib_ma5*0.995) or
                                (ma5>0 and close>=ma5*0.995))
    ut_above_trail = close>0 and (ut_trail<=0 or close>=ut_trail*0.995)
    not_falling = change_pct >= -1.0
    above_short_ma = close>0 and ((ma5>0 and close>=ma5*0.995) or (ma10>0 and close>=ma10*0.995))
    vol_ratio_ok = r.get("vol_ratio",0) >= 1.2
    rising_strong = change_pct >= 1.0
    prev_close=r.get("prev_close",0); prev_ma5=r.get("prev_ma5",0); prev_ma10=r.get("prev_ma10",0)
    below_ma_yesterday = ((prev_ma5>0 and prev_close>0 and prev_close<prev_ma5) or
                          (prev_ma10>0 and prev_close>0 and prev_close<prev_ma10))
    ma_vals=[v for v in (ma5,ma10,ma20,ma60) if v>0]
    ma_loose=True
    if len(ma_vals)>=3 and min(ma_vals)>0:
        cluster=(max(ma_vals)-min(ma_vals))/min(ma_vals)*100
        ma_loose=(cluster>=2.5) or (r.get("vol_ratio",0)>=2.5 and change_pct>=2.0) or below_ma_yesterday
    common = ut_above_trail and not_falling and above_short_ma and vol_ratio_ok and rising_strong and ma_loose
    return {"common":common, "ut":has_ut, "fib":has_fib and fib_above_ma,
            "wr":has_wr, "div":has_div}


def _count_change_signals(sid, ex):
    """條件 G 第3點：『1日前收盤價』突破了幾個變盤訊號壓力區，回傳個數(0~6)，資料不足回 None。
    以 1日前(最近一根完整K) 為基準，壓力K取其前 1/2/3/(4或5)/8/13 根
    （相對今日即 2/3/4/(5或6)/9/14 日前）。
    壓力區：紅K(收≥開)=開盤價；黑K(收<開)=(開+收)/2。突破=1日前收盤 > 壓力區。
    『4或5日前』(相對1日前)任一被突破即算 1 個（總共仍是 6 個變盤訊號）。"""
    try:
        df = fetch_history(sid, ex=ex)
    except Exception:
        return None
    if df is None or len(df) < 14:
        return None
    closes = df["Close"].values.astype(float)
    opens  = (df["Open"] if "Open" in df.columns else df["Close"]).values.astype(float)
    # base：1日前在序列尾端的位置。歷史最後一根若是今日 → 1日前=-2(base=2)，否則=-1(base=1)
    try:
        base = 2 if pd.to_datetime(df.index[-1]).date() == date.today() else 1
    except Exception:
        base = 1
    if len(closes) < base + 13:
        return None
    breaker = closes[-base]
    if breaker <= 0:
        return None
    def resistance(k):            # 相對 1日前往前 k 根（index -(base+k)）
        o = opens[-(base + k)]; c = closes[-(base + k)]
        return o if c >= o else (o + c) / 2.0
    cnt = 0
    for k in (1, 2, 3):
        if breaker > resistance(k): cnt += 1
    if (breaker > resistance(4)) or (breaker > resistance(5)): cnt += 1
    for k in (8, 13):
        if breaker > resistance(k): cnt += 1
    return cnt


@app.route("/api/scan/condition_g", methods=["GET"])
def api_scan_condition_g():
    """條件 G：(UT或費波南) 且 (日威廉/背離/MACD共振/週威廉 任一) + 昨日回檔 + 變盤訊號<4。
    1. (UT∨費波南)∧(日威廉∨指標背離∨MACD共振∨週威廉)，各訊號沿用 A~F 既有門檻。
    2. 1日前回檔：1日前收盤<近3日最高 且 (1日前收盤≤1日前MA5 或 ≤1日前MA10)。
    3. 1日前收盤突破的變盤訊號 < 4 個（壓力K取相對1日前的1/2/3/(4或5)/8/13根，
       即相對今日的2/3/4/(5或6)/9/14日前）。
    流程：先用 cache 過濾(流動性+回檔+群1+共用門檻) → 算 E 命中集合 →
    群2(cache不足才抓週威廉) → 變盤訊號(抓日線歷史)。第一次慢、之後 7 天 cache。"""
    exclude_traditional = request.args.get("exclude_traditional","1") != "0"
    with cache_lock: cached = list(cache.values())
    if not cached:
        return jsonify({"error":"請先執行主掃描","data":[],"hit_count":0,"mode":"G"}), 400

    def _ind_ok(sid):
        if not exclude_traditional: return True
        return not is_excluded_industry(industry_map.get(sid, ""))

    # 先用最便宜的條件過濾：流動性 + 第2點回檔 + 第1點(共用門檻+群1)
    pool = []
    for r in cached:
        if r.get("volume",0) < 20: continue
        if not _ind_ok(r.get("id","")): continue
        if not _f_pullback_ok(r): continue
        st = _g_signal_states(r)
        if not st["common"]: continue
        if not (st["ut"] or st["fib"]): continue
        pool.append((r, st))
    if not pool:
        return jsonify(sanitize({"data":[],"total":len(cached),"hit_count":0,"mode":"G",
                                  "candidates":0,"scanned_at":datetime.now().strftime("%H:%M:%S")}))

    # MACD共振(E)命中集合（群2 來源之一；E 內含週MACD，7天cache）
    try:
        e_hits, _ = scan_condition_e(cached, exclude_traditional=exclude_traditional)
        e_ids = {h.get("id") for h in e_hits}
    except Exception:
        e_ids = set()

    def check(item):
        r, st = item
        sid = r.get("id")
        ex  = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        # 群2：日威廉 / 指標背離 / MACD共振 / 週威廉（前三項用 cache，皆不滿足才抓週威廉）
        g2 = []
        if st["wr"]:     g2.append("日威廉")
        if st["div"]:    g2.append("指標背離")
        if sid in e_ids: g2.append("MACD共振")
        wkw = None
        if not g2:
            wkw = _get_weekly_williams_state(sid, ex, preset="F")
            if wkw.get("wave_complete"): g2.append("週威廉波浪")
        if not g2: return None
        # 第3點：變盤訊號 < 4（以 1日前收盤價為突破者）
        n_sig = _count_change_signals(sid, ex)
        if n_sig is None or n_sig >= 4: return None
        g1 = "UT" if st["ut"] else "費波南"
        r_out = dict(r)
        r_out["buy_mode"]       = "G"
        r_out["change_signals"] = n_sig
        r_out["g1_tag"]         = g1
        r_out["g2_tags"]        = g2
        if wkw: r_out["weekly_wr"] = wkw
        r_out["buy_reasons"]    = [
            f"進場群1：{g1}",
            f"進場群2：{'/'.join(g2)}",
            "1日前回檔確認(昨收<近3日高且≤短均)",
            f"變盤訊號 {n_sig} 個(<4，未追高)",
        ]
        return r_out

    hits = []
    futures = {_global_executor.submit(check, it): it for it in pool}
    for f in as_completed(futures):
        try:
            res = f.result(timeout=30)
            if res: hits.append(res)
        except Exception: pass

    hits.sort(key=lambda x: (x.get("change_signals", 9), -x.get("change_pct", 0)))
    push_scan_summary("G", hits)
    return jsonify(sanitize({"data":hits,"total":len(cached),"hit_count":len(hits),
                              "mode":"G","candidates":len(pool),
                              "scanned_at":datetime.now().strftime("%H:%M:%S")}))


@app.route("/api/status", methods=["GET"])
def api_status():
    now = datetime.now()
    return jsonify({"time":now.strftime("%H:%M:%S"),"date":now.strftime("%Y-%m-%d"),
        "market_open":(9<=now.hour<13) or (now.hour==13 and now.minute<=30),
        "watchlist_count":len(watchlist),"shioaji_ready":_sj_ready,
        "quote_source":"永豐即時" if _sj_ready else "TWSE延遲20分",
        "preload_done":preload_done,"hist_cached":len(hist_cache),
        "full_loaded":full_list_loaded,"scan_cache":len(cache),
        "telegram_configured": bool(TELEGRAM_TOKEN),
        "telegram_enabled": telegram_enabled})

@app.route("/api/telegram/toggle", methods=["POST"])
def api_telegram_toggle():
    global telegram_enabled
    data = request.json or {}
    if "enabled" in data:
        telegram_enabled = bool(data["enabled"])
    return jsonify({"ok": True, "enabled": telegram_enabled,
                    "configured": bool(TELEGRAM_TOKEN)})

@app.route("/api/telegram/test", methods=["POST"])
def api_telegram_test():
    if not TELEGRAM_TOKEN:
        return jsonify({"ok": False, "msg": "未設定 TELEGRAM_TOKEN 環境變數"}), 400
    ok = send_telegram(f"🧪 測試訊息\n時間：{datetime.now().strftime('%H:%M:%S')}")
    return jsonify({"ok": ok, "msg": "已送出" if ok else "送出失敗（看 cmd 錯誤）"})

@app.route("/api/cache/clear", methods=["GET","POST"])
def api_cache_clear():
    global hist_cache
    with hist_cache_lock: hist_cache = {}
    with cache_lock: cache.clear()
    for f in [HIST_CACHE_FILE, SCAN_CACHE_FILE]:
        try:
            if os.path.exists(f): os.remove(f)
        except: pass
    return jsonify({"ok":True,"msg":"快取已清除"})

@app.route("/api/stocklist/status", methods=["GET"])
def api_stocklist_status():
    return jsonify({"full_loaded":full_list_loaded,"total_stocks":len(watchlist),
        "cache_count":len(cache),"hist_cached":len(hist_cache)})

@app.route("/api/scheduler/start", methods=["POST"])
def api_scheduler_start():
    global scheduler_running, scheduler_interval
    data = request.json or {}
    scheduler_interval = int(data.get("interval",30))
    if scheduler_running: return jsonify({"ok":False,"msg":"排程已在執行"})
    scheduler_running = True
    threading.Thread(target=scheduler_loop, daemon=True).start()
    send_telegram(f"✅ <b>選股系統已啟動</b>\n⏱ 每{scheduler_interval}分鐘掃描\n📊 監控{len(watchlist)}支")
    return jsonify({"ok":True,"msg":f"排程已啟動，每{scheduler_interval}分鐘"})

@app.route("/api/scheduler/stop", methods=["POST"])
def api_scheduler_stop():
    global scheduler_running
    scheduler_running = False
    return jsonify({"ok":True,"msg":"排程已停止"})

@app.route("/api/scheduler/status", methods=["GET"])
def api_scheduler_status():
    return jsonify({"running":scheduler_running,"interval":scheduler_interval})

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.json
    sid  = data.get("stock_id","2330").strip(); ex = data.get("ex","tse")
    strategy_ids = data.get("strategies",["S1","S2","S3","S4","S5","S6"])
    hold_days = int(data.get("hold_days",10)); stop_pct = float(data.get("stop_pct",0.07))
    try:
        df = fetch_history(sid, period="1y", ex=ex)
        if len(df) < 60: return jsonify({"error":f"{sid} 歷史資料不足"}), 400
        closes = df["Close"]; volumes = df["Volume"]
        highs  = df["High"] if "High" in df.columns else closes
        lows   = df["Low"]  if "Low"  in df.columns else closes
        all_trades = []; strat_summary = []
        for strat_id in strategy_ids:
            sig_arr = []
            for i in range(65, len(closes)):
                cs = closes.iloc[:i+1]; vs = volumes.iloc[:i+1]
                hs = highs.iloc[:i+1];  ls = lows.iloc[:i+1]
                ind_i = calc_all_indicators(cs, hs, ls, vs)
                triggered = eval_strategies(ind_i, cs, vs)
                sig_arr.append(any(t["id"]==strat_id for t in triggered))
            trades = []; in_t = False; ep = 0.0; ed = None; dh = 0
            px = closes.values; dt = closes.index; so = 65
            for i in range(so, len(px)):
                si = i - so
                if in_t:
                    dh += 1
                    if px[i] <= ep*(1-stop_pct):
                        pnl = (px[i]-ep)/ep*100
                        trades.append({"策略":strat_id,"進場日":str(ed.date()),"出場日":str(dt[i].date()),
                            "進場價":round(ep,2),"出場價":round(float(px[i]),2),
                            "報酬率%":round(pnl,2),"出場原因":"停損","持有天數":dh})
                        in_t = False
                    elif dh >= hold_days:
                        pnl = (px[i]-ep)/ep*100
                        trades.append({"策略":strat_id,"進場日":str(ed.date()),"出場日":str(dt[i].date()),
                            "進場價":round(ep,2),"出場價":round(float(px[i]),2),
                            "報酬率%":round(pnl,2),"出場原因":"到期","持有天數":dh})
                        in_t = False
                elif si < len(sig_arr) and sig_arr[si]:
                    in_t = True; ep = float(px[i]); ed = dt[i]; dh = 0
            if not trades:
                strat_summary.append({"策略":strat_id,"交易次數":0,"勝率%":0,"平均報酬%":0,
                    "最大獲利%":0,"最大虧損%":0,"總報酬%":0,"盈虧比":0})
                continue
            all_trades.extend(trades)
            wins   = [t for t in trades if t["報酬率%"]>0]
            losses = [t for t in trades if t["報酬率%"]<=0]
            aw = sum(t["報酬率%"] for t in wins)/len(wins)     if wins   else 0
            al = sum(t["報酬率%"] for t in losses)/len(losses) if losses else 0
            strat_summary.append({"策略":strat_id,"交易次數":len(trades),
                "勝率%":round(len(wins)/len(trades)*100,1),
                "平均報酬%":round(sum(t["報酬率%"] for t in trades)/len(trades),2),
                "最大獲利%":round(max(t["報酬率%"] for t in trades),2),
                "最大虧損%":round(min(t["報酬率%"] for t in trades),2),
                "總報酬%":round(sum(t["報酬率%"] for t in trades),2),
                "盈虧比":round(abs(aw/al),2) if al!=0 else 0})
        return jsonify({"stock_id":sid,"hold_days":hold_days,"stop_pct":stop_pct*100,
            "period":"1年","summary":strat_summary,"trades":all_trades,"total_trades":len(all_trades)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/why/<sid>", methods=["GET"])
def api_why(sid):
    """診斷某支股票為何沒進條件 A/B/C，回傳每個閘門的真假值"""
    mode = request.args.get("mode","C").upper()
    with cache_lock: r = cache.get(sid)
    if not r:
        return jsonify({"error":f"{sid} 不在 cache，請先跑掃描","sid":sid}), 404
    ind = industry_map.get(sid, "(未知)")
    industry_excluded = is_excluded_industry(ind)
    has_fib = bool(r.get("fib",{}).get("buy_signals"))
    has_ut  = bool(r.get("ut_buy"))
    has_wr  = bool(r.get("wr",{}).get("buy_signals"))
    div     = r.get("div",{})
    has_div = bool(div.get("buy_signals")) or div.get("score",0) >= 2
    close      = r.get("close",0)
    ma5  = r.get("ma5",0);  ma10 = r.get("ma10",0)
    ma20 = r.get("ma20",0); ma60 = r.get("ma60",0)
    change_pct = r.get("change_pct",0)
    vol        = r.get("volume",0)
    vol_ratio  = r.get("vol_ratio",0)
    ut_trail   = r.get("ut_trail",0)
    confirm_score = (r.get("confirm") or {}).get("score",0)
    fib_d = r.get("fib",{}).get("details",{})
    fib_ma3 = fib_d.get("ma3", ma5); fib_ma5 = fib_d.get("ma5", ma5)
    fib_above_ma = close>0 and ((fib_ma3>0 and close>=fib_ma3*0.995) or
                                (fib_ma5>0 and close>=fib_ma5*0.995) or
                                (ma5>0 and close>=ma5*0.995))
    ut_above_trail = close>0 and (ut_trail<=0 or close>=ut_trail*0.995)
    not_falling = change_pct >= -1.0
    above_short_ma = close>0 and ((ma5>0 and close>=ma5*0.995) or
                                   (ma10>0 and close>=ma10*0.995))
    vol_ratio_ok = vol_ratio >= 1.2
    rising_strong = change_pct >= 1.0
    prev_close = r.get("prev_close", 0)
    prev_ma5   = r.get("prev_ma5", 0)
    prev_ma10  = r.get("prev_ma10", 0)
    below_ma_yesterday = (
        (prev_ma5  > 0 and prev_close > 0 and prev_close < prev_ma5) or
        (prev_ma10 > 0 and prev_close > 0 and prev_close < prev_ma10)
    )
    ma_vals = [v for v in (ma5,ma10,ma20,ma60) if v>0]
    ma_loose = True; ma_cluster_pct = None
    if len(ma_vals) >= 3 and min(ma_vals)>0:
        ma_cluster_pct = (max(ma_vals)-min(ma_vals))/min(ma_vals)*100
        ma_loose = (ma_cluster_pct >= 2.5) or (vol_ratio >= 2.5 and change_pct >= 2.0) or below_ma_yesterday
    ma_gate_name = (f"MA偏差≥2.5%或量價共振或昨日在均線下（實際偏差{ma_cluster_pct:.2f}%, 昨收{prev_close} vs 昨MA5={prev_ma5:.2f}/MA10={prev_ma10:.2f}）"
                    if ma_cluster_pct else "MA偏差≥2.5%或量價共振或昨日在均線下")
    common = {
        "成交量≥20張":  vol >= 20,
        "排除產業":      not industry_excluded,
        "UT在追蹤線上":  ut_above_trail,
        "未跌≥1%":       not_falling,
        "現價在短MA上":  above_short_ma,
        "量比≥1.2":      vol_ratio_ok,
        "漲幅≥1%":       rising_strong,
        ma_gate_name:    ma_loose,
    }
    if mode == "A":
        gates = {**common, "has_UT":has_ut, "has_威廉":has_wr, "has_背離":has_div}
    elif mode == "B":
        ex_b = next((s.get("ex","tse") for s in watchlist if s["id"]==sid), "tse")
        wkw_b = _get_weekly_williams_state(sid, ex_b, preset="B")
        gates = {**common, "has_UT":has_ut, "has_費波南":has_fib, "費波南在MA上":fib_above_ma,
                 "週線威廉波浪完成": bool(wkw_b.get("wave_complete"))}
    else:  # C
        gates = {**common, "has_UT":has_ut, "has_費波南":has_fib, "has_威廉":has_wr,
                 "費波南在MA上":fib_above_ma,
                 f"確認分>4（實際{confirm_score}）": confirm_score > 4}
    passed = all(gates.values())
    fails = {k:v for k,v in gates.items() if not v}
    return jsonify({"sid":sid, "name":r.get("name",""), "industry":ind,
        "mode":mode, "passed": passed,
        "values":{
            "close":close,"change_pct":round(change_pct,2),"volume":vol,
            "vol_ratio":round(vol_ratio,2),"ma5":ma5,"ma10":ma10,"ma20":ma20,"ma60":ma60,
            "ma_cluster_pct":round(ma_cluster_pct,2) if ma_cluster_pct else None,
            "ut_trail":ut_trail,"confirm_score":confirm_score,
        },
        "gates": gates, "fails": fails})


@app.route("/api/debug/<sid>", methods=["GET"])
def api_debug(sid):
    ex = request.args.get("ex","tse")
    rt = fetch_realtime_price(sid, ex)
    hist = fetch_history(sid, ex=ex)
    return jsonify({"rt":rt,"has_history":len(hist)>=30,"hist_rows":len(hist),
        "hist_last_date":str(hist.index[-1].date()) if len(hist)>0 else None,
        "hist_cached":bool(_get_hist_cache(_hist_key(sid,"3mo",ex)))})

@app.route("/api/export/csv", methods=["POST"])
def api_export_csv():
    """前端把當前顯示的列表 POST 過來，回傳 CSV 檔案"""
    from flask import Response
    import csv, io
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows", [])
    mode = payload.get("mode", "X")
    if not rows:
        return jsonify({"error":"沒有資料"}), 400
    buf = io.StringIO()
    buf.write("﻿")  # BOM for Excel UTF-8
    w = csv.writer(buf)
    if mode == "E":
        # 條件 E 專屬：優先分排序 + 進場位階 + 週MACD狀態 + 共振品質，方便不看圖直接決定買進順序
        w.writerow(["#","股號","名稱","現價","漲跌幅%","量比","進場優先分","E評分",
                    "位階","離MA20%","週MACD狀態","日底背離","共振品質","MA20","MA60"])
        for i, r in enumerate(rows, 1):
            wk  = r.get("weekly_macd") or {}
            div = r.get("div") or {}
            weekly_txt = "零軸上" if wk.get("above_zero") else "零軸下"
            if wk.get("golden_cross"): weekly_txt += "/剛金叉"
            if wk.get("hist_rising"):  weekly_txt += "/紅柱放大"
            base = {"日週MACD共振","週MACD多頭","日MACD金叉","站上MA20"}
            flags = [x for x in (r.get("buy_reasons") or []) if x not in base]
            w.writerow([i, r.get("id",""), r.get("name",""),
                        r.get("close",0), r.get("change_pct",0), r.get("vol_ratio",0),
                        r.get("prio",0), r.get("e_score",0),
                        r.get("level_tag",""), r.get("bias20",""),
                        weekly_txt, div.get("score",0), " ".join(flags),
                        r.get("ma20",0), r.get("ma60",0)])
        csv_data = buf.getvalue()
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return Response(csv_data, mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scan_E_{ts}.csv"})

    if mode == "F":
        # 條件 F 專屬：直接看出每檔憑什麼進 F（符合哪個條件 + 週威廉波浪 + 昨日回檔數據）
        w.writerow(["#","股號","名稱","現價","漲跌幅%","量比","符合條件",
                    "週威廉深底WR","波峰數","距今(週)","週WR現值",
                    "昨收","近3日高","昨MA5","昨MA10","MA20","MA60"])
        for i, r in enumerate(rows, 1):
            wk = r.get("weekly_wr") or {}
            w.writerow([i, r.get("id",""), r.get("name",""),
                        r.get("close",0), r.get("change_pct",0), r.get("vol_ratio",0),
                        "/".join(r.get("f_tags", [])),
                        wk.get("deep_low_val",""), wk.get("peak_count",""),
                        wk.get("bars_from_low",""), wk.get("wr_now",""),
                        r.get("prev_close",0), r.get("prev3_high",0),
                        r.get("prev_ma5",0), r.get("prev_ma10",0),
                        r.get("ma20",0), r.get("ma60",0)])
        csv_data = buf.getvalue()
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return Response(csv_data, mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scan_F_{ts}.csv"})

    w.writerow(["#","股號","名稱","現價","漲跌","漲跌幅%","成交量(張)","量比",
                "MA5","MA10","MA20","MA60","背離分數","週背離分數","來源","訊號"])
    for i, r in enumerate(rows, 1):
        div = r.get("div") or {}
        sigs = []
        if r.get("ut_buy"): sigs.append("UT")
        if (r.get("fib") or {}).get("buy_signals"): sigs.append("費波南")
        if (r.get("wr")  or {}).get("buy_signals"): sigs.append("威廉波浪")
        if div.get("buy_signals") or div.get("score",0)>=2: sigs.append(f"背離{div.get('score',0)}")
        source = "DB延遲確認" if r.get("db_delayed") else "即時"
        w.writerow([i, r.get("id",""), r.get("name",""),
                    r.get("close",0), r.get("change",0), r.get("change_pct",0),
                    r.get("volume",0), r.get("vol_ratio",0),
                    r.get("ma5",0), r.get("ma10",0), r.get("ma20",0), r.get("ma60",0),
                    div.get("score",0), r.get("weekly_score",""), source,
                    " ".join(sigs)])
    csv_data = buf.getvalue()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"scan_{mode}_{ts}.csv"
    return Response(csv_data, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/")
def index():
    try:
        return render_template_string(open("index.html", encoding="utf-8").read())
    except:
        return "<h2>index.html 不存在</h2>", 404


# ══════════════════════════════════════════════════════════
# ▌ 啟動序列
# ══════════════════════════════════════════════════════════
def startup_sequence():
    print("  🚀 啟動序列開始...")
    fast_warmup()                  # ★ numba 預編譯（含 calc_all_indicators_fast）
    _load_delisted()
    load_full_stock_list()
    fetch_industry_map()
    _init_div_db()
    warm_up_cache()
    _load_scan_cache_disk()        # ★ 載入上次掃描結果，第一次掃描即可走 fast mode
    _save_delisted()
    print("  🎉 啟動完成！可以開始掃描。")

threading.Thread(target=startup_sequence, daemon=True).start()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║   盤中選股系統（極速版 v2）啟動中...                 ║
║   瀏覽器開啟：http://localhost:5000                  ║
║                                                      ║
║   優化新增：                                         ║
║   ✅ numba JIT（UT Bot/Pivot 加速 10-50x）           ║
║   ✅ 預擷取 numpy（去除 .iloc 開銷）                 ║
║   ✅ 移除 pd.concat 單列附加                         ║
║   ✅ 批次平行下載（TSE+OTC 混合）                    ║
║   ✅ 主掃描預先補齊歷史                              ║
╚══════════════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=5000, debug=False)
