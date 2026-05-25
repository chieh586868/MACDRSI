"""
================================================================
指標背離選股系統 v2 - 日K / 週K 雙模式（極速版）
================================================================
  🔵 日K背離：掃描日線 MACD/RSI/KD 底背離
  🟡 週K背離：掃描週線 MACD/RSI/KD 底背離（更強訊號）

加速重點（vs 原版）：
  ★ pivot 偵測改用 numba JIT（fast_indicators），加速 20-50x
  ★ calc_ind 預擷取 numpy，避免重複 .iloc
  ★ batch_preload 改為「全部批次平行」而非 TSE→OTC 分組序列
  ★ resample 週K 結果快取，避免日週雙重掃描各跑一次
  ★ analyze 路徑減少 DataFrame copy 與 list.append

啟動：python divergence_screener.py
瀏覽：http://localhost:5001
================================================================
"""
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests as req
import json, time, threading, os, warnings, logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
import yfinance as yf

from fast_indicators import (
    find_pivot_lows_nb,
    calc_indicators_fast,
    warmup as fast_warmup,
)
from williams_v4 import calc_williams_signals_v4

app = Flask(__name__)
CORS(app)
app.config["JSON_SORT_KEYS"] = False

# ── 設定 ──────────────────────────────────────────────────
PORT             = 5001
SCAN_WORKERS     = 8     # ★ 4 核心 CPU 最佳值約 6-8；過多反而 GIL 競爭
YF_PARALLEL_BATCHES = 6           # ★ 預載時並行批次數
HIST_TTL         = 86400 * 3
FULL_LIST_FILE   = "tw_stocks.json"
DELISTED_FILE    = "delisted_cache.json"
HIST_CACHE_FILE  = "hist_cache.pkl"

# ── 全局狀態 ──────────────────────────────────────────────
watchlist     = []
hist_cache    = {}
hist_lock     = threading.Lock()
_delisted_set = set()
_del_lock     = threading.Lock()
preload_done  = False
_executor     = ThreadPoolExecutor(max_workers=SCAN_WORKERS)
_dl_executor  = ThreadPoolExecutor(max_workers=YF_PARALLEL_BATCHES * 2)
scan_progress = {"done": 0, "total": 0, "running": False, "mode": ""}

# ★ 週K resample 快取（避免日週雙重掃描各 resample 一次）
_wk_cache      = {}
_wk_cache_lock = threading.Lock()


def sanitize(obj):
    """非遞迴清洗"""
    if isinstance(obj, (dict, list)):
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
            else:
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


# ── 歷史資料 ──────────────────────────────────────────────
def get_hist(sid, ex="tse", period="3mo", interval="1d"):
    with _del_lock:
        if sid in _delisted_set:
            return pd.DataFrame()
    key = f"{sid}_{period}_{interval}_{ex}"
    with hist_lock:
        entry = hist_cache.get(key)
    if entry and (time.time() - entry["ts"]) < HIST_TTL:
        return entry["df"]
    suffixes = [".TWO", ".TW"] if ex == "otc" else [".TW", ".TWO"]
    for suf in suffixes:
        try:
            df = yf.download(f"{sid}{suf}", period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            min_bars = 20 if interval == "1wk" else 30
            if len(df) >= min_bars:
                with hist_lock:
                    hist_cache[key] = {"df": df, "ts": time.time()}
                return df
        except Exception:
            continue
    with _del_lock:
        _delisted_set.add(sid)
    return pd.DataFrame()


# ── 磁碟快取 ──────────────────────────────────────────────
def load_hist_cache_disk():
    global hist_cache
    try:
        import pickle
        if not os.path.exists(HIST_CACHE_FILE):
            return False
        if (time.time() - os.path.getmtime(HIST_CACHE_FILE)) >= HIST_TTL:
            return False
        with open(HIST_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if len(data) < 100:
            return False
        with hist_lock:
            hist_cache.update(data)
        print(f"  ✅ 磁碟快取載入：{len(hist_cache)} 筆（與 macd_app 共用）")
        return True
    except Exception as e:
        print(f"  ⚠ 快取載入失敗: {e}")
        return False


def save_hist_cache_disk():
    try:
        import pickle
        with hist_lock:
            data = dict(hist_cache)
        if len(data) < 50:
            return
        with open(HIST_CACHE_FILE, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  💾 快取存檔：{len(data)} 筆")
    except Exception as e:
        print(f"  ⚠ 快取存檔失敗: {e}")


def _batch_download(stocks: list, period: str, label: str = "") -> int:
    """
    對指定期間批次下載歷史，跳過已快取者。
    被 batch_preload(3mo) 與 ensure_1y_cached() 共用。
    """
    if not stocks:
        return 0

    # 找出真正缺漏的
    missing = []
    for s in stocks:
        key = f"{s['id']}_{period}_1d_{s.get('ex','tse')}"
        with hist_lock:
            entry = hist_cache.get(key)
        if entry is not None and (time.time() - entry["ts"]) < HIST_TTL:
            continue
        with _del_lock:
            if s["id"] in _delisted_set:
                continue
        missing.append(s)

    if not missing:
        return 0

    total = len(missing)
    lbl = label or period
    print(f"  🔄 批次下載 {lbl} 歷史：{total} 支...")
    t0 = time.time()
    loaded = [0]
    lock   = threading.Lock()
    min_bars = 60 if period == "1y" else 25

    def dl_batch(batch, suffix, ex_name):
        if not batch:
            return
        tickers = " ".join(f"{s['id']}{suffix}" for s in batch)
        try:
            raw = yf.download(tickers, period=period, group_by="ticker",
                              progress=False, auto_adjust=True, threads=True)
            single = len(batch) == 1
            cols0 = None if single else raw.columns.get_level_values(0)
            for s in batch:
                tid = f"{s['id']}{suffix}"
                try:
                    if single:
                        df = raw
                    elif tid in cols0:
                        df = raw[tid]
                    else:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna()
                    if len(df) >= min_bars:
                        key = f"{s['id']}_{period}_1d_{ex_name}"
                        with hist_lock:
                            hist_cache[key] = {"df": df, "ts": time.time()}
                        with lock:
                            loaded[0] += 1
                except Exception:
                    pass
        except Exception:
            for s in batch:
                df = get_hist(s["id"], s.get("ex", "tse"), period=period)
                if not df.empty:
                    with lock:
                        loaded[0] += 1

    tse = [s for s in missing if s.get("ex", "tse") == "tse"]
    otc = [s for s in missing if s.get("ex", "tse") == "otc"]
    BSIZ = 40
    tasks = []
    for i in range(0, len(tse), BSIZ):
        tasks.append(_dl_executor.submit(dl_batch, tse[i:i+BSIZ], ".TW",  "tse"))
    for i in range(0, len(otc), BSIZ):
        tasks.append(_dl_executor.submit(dl_batch, otc[i:i+BSIZ], ".TWO", "otc"))

    done = 0
    for f in as_completed(tasks):
        done += 1
        print(f"  批次 {done}/{len(tasks)} 完成（已載入 {loaded[0]} 支）...", end="\r")

    elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ {lbl} 載入完成：{loaded[0]}/{total} 支，{elapsed} 秒")
    return loaded[0]


# ── 1y 快取補齊（dw 掃描需要）────────────────────────────
_1y_loaded = False
_1y_lock   = threading.Lock()

def ensure_1y_cached():
    """確保所有股票的 1y 歷史已快取；dw 掃描前呼叫"""
    global _1y_loaded
    with _1y_lock:
        if _1y_loaded:
            return
        loaded = _batch_download(list(watchlist), period="1y", label="1y")
        if loaded > 0:
            save_hist_cache_disk()
        _1y_loaded = True


def batch_preload():
    """★ 啟動時預載 3mo 歷史（dw 模式需要的 1y 另外懶載入）"""
    global preload_done, _1y_loaded
    if load_hist_cache_disk():
        preload_done = True
        # 磁碟快取若含 1y，跳過二次下載
        with hist_lock:
            if any(k.endswith("_1y_1d_tse") or k.endswith("_1y_1d_otc")
                   for k in hist_cache):
                _1y_loaded = True
        print(f"  🎉 就緒（快取命中）！{len(watchlist)} 支股票可掃描")
        return

    _batch_download(list(watchlist), period="3mo", label="3mo")
    save_hist_cache_disk()
    preload_done = True
    print(f"  🎉 就緒！{len(watchlist)} 支股票可掃描")


# ── 增量更新：自動偵測快取是否需要補抓最近交易日 ──────────
def latest_trading_day_tw() -> "pd.Timestamp":
    """
    回傳台股最新交易日（以本機時間判斷，假設使用者在台灣時區）。
    - 週末 → 上週五
    - 平日盤中（13:30 前）→ 昨日
    - 平日盤後（13:30 後）→ 今日
    """
    now = datetime.now()
    d = now.date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    if d == now.date():
        cur_min = now.hour * 60 + now.minute
        if cur_min < 13 * 60 + 30:
            d -= timedelta(days=1)
            while d.weekday() >= 5:
                d -= timedelta(days=1)
    return pd.Timestamp(d)


def _incremental_download(stocks: list, period: str = "10d") -> int:
    """批次下載最近 N 天並合併進現有 3mo / 1y 快取"""
    if not stocks:
        return 0
    t0 = time.time()
    merged = [0]
    lock = threading.Lock()

    def dl_batch(batch, suffix, ex_name):
        if not batch:
            return
        tickers = " ".join(f"{s['id']}{suffix}" for s in batch)
        try:
            raw = yf.download(tickers, period=period, group_by="ticker",
                              progress=False, auto_adjust=True, threads=True)
        except Exception:
            return
        single = len(batch) == 1
        try:
            cols0 = None if single else raw.columns.get_level_values(0)
        except Exception:
            return
        for s in batch:
            tid = f"{s['id']}{suffix}"
            try:
                if single:
                    new_df = raw
                elif tid in cols0:
                    new_df = raw[tid]
                else:
                    continue
                if isinstance(new_df.columns, pd.MultiIndex):
                    new_df.columns = new_df.columns.get_level_values(0)
                new_df = new_df.dropna()
                if len(new_df) == 0:
                    continue
                sid = s["id"]
                cutoff = new_df.index[0]
                # 合併到同股票的所有期間快取
                for hist_period in ("3mo", "1y"):
                    key = f"{sid}_{hist_period}_1d_{ex_name}"
                    with hist_lock:
                        old_entry = hist_cache.get(key)
                    if old_entry is None:
                        continue
                    old_df = old_entry["df"]
                    head = old_df[old_df.index < cutoff]
                    combined = pd.concat([head, new_df])
                    # 修剪：3mo 保留 ~90 row, 1y 保留 ~280 row
                    max_rows = 280 if hist_period == "1y" else 90
                    if len(combined) > max_rows:
                        combined = combined.tail(max_rows)
                    with hist_lock:
                        hist_cache[key] = {"df": combined, "ts": time.time()}
                    with lock:
                        merged[0] += 1
            except Exception:
                pass

    tse = [s for s in stocks if s.get("ex", "tse") == "tse"]
    otc = [s for s in stocks if s.get("ex", "tse") == "otc"]
    BSIZ = 50
    tasks = []
    for i in range(0, len(tse), BSIZ):
        tasks.append(_dl_executor.submit(dl_batch, tse[i:i+BSIZ], ".TW",  "tse"))
    for i in range(0, len(otc), BSIZ):
        tasks.append(_dl_executor.submit(dl_batch, otc[i:i+BSIZ], ".TWO", "otc"))
    done = 0
    for f in as_completed(tasks):
        done += 1
        print(f"  增量批次 {done}/{len(tasks)} 完成（已合併 {merged[0]} 筆）...", end="\r")

    elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ 增量更新完成：合併 {merged[0]} 筆快取，{elapsed} 秒")
    return merged[0]


def incremental_refresh():
    """
    啟動時呼叫：檢查快取最後一根 K 棒是否為最新交易日。
    若否，批次下載最近 10 天並合併進快取。比手動 del hist_cache.pkl
    再重抓 ~30 秒快，因為只抓 10 天而不是 1 年。
    """
    if not hist_cache:
        return

    # 找快取裡的最新日期（掃前幾筆即可，假設大部分股票同步）
    latest = None
    checked = 0
    for k, entry in hist_cache.items():
        df = entry["df"]
        if df is None or len(df) == 0:
            continue
        last = df.index[-1]
        if latest is None or last > latest:
            latest = last
        checked += 1
        if checked >= 20:
            break

    if latest is None:
        return

    expected = latest_trading_day_tw()
    latest_date = latest.date() if hasattr(latest, "date") else latest
    expected_date = expected.date()

    if latest_date >= expected_date:
        print(f"  ✅ 快取已是最新交易日（{latest_date}）")
        return

    print(f"  📅 快取最後 K 棒: {latest_date} → 最新交易日: {expected_date}")
    print(f"  🔄 批次補抓最近 10 天...")
    n = _incremental_download(list(watchlist), period="10d")
    if n > 0:
        save_hist_cache_disk()


def load_delisted():
    global _delisted_set
    try:
        if os.path.exists(DELISTED_FILE):
            with open(DELISTED_FILE, "r") as f:
                _delisted_set = set(json.load(f))
            print(f"  📋 已知下市：{len(_delisted_set)} 支")
    except Exception:
        pass


def save_delisted():
    try:
        with open(DELISTED_FILE, "w") as f:
            json.dump(list(_delisted_set), f)
    except Exception:
        pass


def load_stock_list():
    global watchlist
    if os.path.exists(FULL_LIST_FILE):
        try:
            if (time.time() - os.path.getmtime(FULL_LIST_FILE)) < 86400:
                with open(FULL_LIST_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data) > 100:
                    watchlist = data
                    print(f"  ✅ 股票清單：{len(data)} 支")
                    return
        except Exception:
            pass
    print("  🌐 下載台股清單...")
    tse, otc = [], []
    try:
        r = req.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                    timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        tse = [{"id": str(x["Code"]), "name": str(x["Name"]), "ex": "tse"}
               for x in r.json() if str(x.get("Code", "")).isdigit() and len(str(x["Code"])) == 4]
    except Exception:
        pass
    try:
        r = req.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
                    timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        otc = [{"id": str(x["SecuritiesCompanyCode"]), "name": str(x["CompanyName"]), "ex": "otc"}
               for x in r.json() if str(x.get("SecuritiesCompanyCode", "")).isdigit()
               and len(str(x["SecuritiesCompanyCode"])) == 4]
    except Exception:
        pass
    seen = set(); unique = []
    for s in tse + otc:
        if s["id"] not in seen:
            seen.add(s["id"]); unique.append(s)
    if len(unique) > 100:
        watchlist = unique
        with open(FULL_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(unique, f, ensure_ascii=False)
        print(f"  ✅ {len(unique)} 支（上市{len(tse)}+上櫃{len(otc)}）")


# ── 指標計算（★ 改用 numba 全部 inline 計算，去除 pandas 開銷）
def calc_ind(closes, highs, lows, volumes):
    """
    用 fast_indicators.calc_indicators_fast 一次算完所有指標。
    輸出仍包成 pd.Series 維持與舊版相容（避免動到 detect_div 等下游）。
    profile 顯示原版 pandas 版本 1y 資料約 50ms/股，
    改 numba 後預期 1-2ms/股（25-50x 加速）。
    """
    c_arr = np.asarray(closes, dtype=np.float64)
    h_arr = np.asarray(highs,  dtype=np.float64)
    l_arr = np.asarray(lows,   dtype=np.float64)
    v_arr = np.asarray(volumes, dtype=np.float64)

    dif, dea, hist, rsi, K, D, ma5, ma10, ma20, vol_ratio = \
        calc_indicators_fast(c_arr, h_arr, l_arr, v_arr)

    idx = closes.index if hasattr(closes, "index") else None
    return {
        "dif":       pd.Series(dif,       index=idx),
        "dea":       pd.Series(dea,       index=idx),
        "hist":      pd.Series(hist,      index=idx),
        "rsi":       pd.Series(rsi,       index=idx),
        "K":         pd.Series(K,         index=idx),
        "D":         pd.Series(D,         index=idx),
        "ma5":       pd.Series(ma5,       index=idx),
        "ma10":      pd.Series(ma10,      index=idx),
        "ma20":      pd.Series(ma20,      index=idx),
        "vol_ratio": pd.Series(vol_ratio, index=idx),
    }


# ── 背離偵測（★ pivot 使用 numba；★ 跳過無必要的 NaN 填補）──
def detect_div(price, indicator, lookback=120, min_gap=4, max_bars=15):
    r = {"regular_bull": False, "hidden_bull": False, "detail": ""}
    n_total = len(price)
    lb = lookback if lookback < n_total else n_total
    if lb < 15:
        return r

    # ★ 直接拿 numpy view，跳過 pandas .iloc 開銷
    pv_full = price.values
    iv_full = indicator.values
    pv = pv_full[-lb:]
    iv = iv_full[-lb:]

    # ★ 多數股票沒 NaN，先檢查再決定要不要做 ffill/bfill/fillna
    if np.isnan(pv).any():
        pv = pd.Series(pv).ffill().bfill().fillna(0).values
    if np.isnan(iv).any():
        iv = pd.Series(iv).ffill().bfill().fillna(0).values

    # 確保是 float64（numba 要求）
    if pv.dtype != np.float64:
        pv = pv.astype(np.float64)
    if iv.dtype != np.float64:
        iv = iv.astype(np.float64)

    pp = find_pivot_lows_nb(pv, 3, 3)
    ip = find_pivot_lows_nb(iv, 3, 3)
    if len(pp) < 2 or len(ip) < 2:
        return r
    n = len(pv)
    p1 = int(pp[-1])

    # 早退：如果最新 pivot 太遠就不用算
    bars = n - 1 - p1
    if bars > max_bars:
        return r

    p2 = -1
    for k in range(len(pp) - 2, -1, -1):
        if p1 - pp[k] >= min_gap:
            p2 = int(pp[k]); break
    if p2 < 0:
        return r

    sr = 6
    # 不轉 ip.tolist()，直接在 numpy array 上找 nearest
    def ni(pi):
        best = pi
        best_d = sr + 1
        for x in ip:
            d = x - pi if x >= pi else pi - x
            if d <= sr and d < best_d:
                best = int(x); best_d = d
        return best
    i1, i2 = ni(p1), ni(p2)
    vp1, vp2 = float(pv[p1]), float(pv[p2])
    vi1, vi2 = float(iv[i1]), float(iv[i2])
    if vp1 != vp1 or vp2 != vp2 or vi1 != vi1 or vi2 != vi2:
        return r
    if vp1 < vp2 and vi1 > vi2:
        r["regular_bull"] = True
        r["detail"] = f"{vp2:.2f}→{vp1:.2f}（底部抬高，距今{bars}根）"
    if vp1 > vp2 and vi1 < vi2:
        r["hidden_bull"] = True
        r["detail"] = f"{vp2:.2f}→{vp1:.2f}（更高低點，距今{bars}根）"
    return r


def calc_div_score(ind, closes, highs, lows, volumes, max_bars=15,
                   use_macd=True, use_rsi=True, use_kd=True):
    sigs = []; score = 0
    pool = []
    if use_macd: pool.append(("hist", "MACD", "#CE93D8", "#80DEEA", 4))
    if use_rsi:  pool.append(("rsi",  "RSI",  "#A5D6A7", "#80CBC4", 3))
    if use_kd:   pool.append(("K",    "KD",   "#FFCC80", "#FFAB91", 2))
    for key, name, cr, ch, pr in pool:
        try:
            d = detect_div(closes, ind[key], max_bars=max_bars)
            if d["regular_bull"]:
                score += 2
                sigs.append({"id": f"{name}正規", "name": f"{name}正規底背離",
                    "color": cr, "desc": d["detail"], "type": "regular",
                    "indicator": name, "priority": pr})
            if d["hidden_bull"]:
                score += 1
                sigs.append({"id": f"{name}隱藏", "name": f"{name}隱藏底背離",
                    "color": ch, "desc": d["detail"], "type": "hidden",
                    "indicator": name, "priority": pr - 1})
        except Exception:
            pass
    reg = sum(1 for s in sigs if s["type"] == "regular")
    hid = sum(1 for s in sigs if s["type"] == "hidden")
    if reg >= 2: score += 1
    if hid >= 2: score += 1
    sigs.sort(key=lambda x: -x["priority"])
    parts = []
    if reg: parts.append("正規:" + "+".join(s["indicator"] for s in sigs if s["type"] == "regular"))
    if hid: parts.append("隱藏:" + "+".join(s["indicator"] for s in sigs if s["type"] == "hidden"))
    return {"signals": sigs, "score": min(score, 7),
            "summary": "；".join(parts) if parts else "無",
            "regular_count": reg, "hidden_count": hid}


# ── 週K resample（含快取）─────────────────────────────────
def _get_weekly(sid: str, df_daily: pd.DataFrame) -> pd.DataFrame:
    """同一支股票的週K resample 結果快取，三指標/日週雙掃時受惠"""
    key = sid
    with _wk_cache_lock:
        entry = _wk_cache.get(key)
    if entry and entry["len"] == len(df_daily):
        return entry["wdf"]
    df_daily.index = pd.to_datetime(df_daily.index)
    wdf = df_daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    with _wk_cache_lock:
        _wk_cache[key] = {"wdf": wdf, "len": len(df_daily)}
    return wdf


# ── 單股分析（日K / 週K / KD）─────────────────────────────
def _last_float(arr, default=0.0):
    v = arr[-1] if len(arr) else default
    return default if v != v else float(v)


def analyze_one(stock, mode="daily"):
    sid = stock["id"]; ex = stock.get("ex", "tse")
    try:
        if mode == "weekly":
            df_d = get_hist(sid, ex, period="1y", interval="1d")
            if len(df_d) < 20:
                return None
            df = _get_weekly(sid, df_d)
            if len(df) < 12:
                return None
            max_bars = 10
        else:
            df = get_hist(sid, ex, period="3mo", interval="1d")
            if len(df) < 30:
                return None
            max_bars = 15

        closes  = df["Close"]
        highs   = df["High"]   if "High"   in df.columns else closes
        lows    = df["Low"]    if "Low"    in df.columns else closes
        volumes = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=closes.index)

        ind = calc_ind(closes, highs, lows, volumes)
        use_macd = (mode != "kd")
        use_rsi  = (mode != "kd")
        use_kd   = True
        div = calc_div_score(ind, closes, highs, lows, volumes, max_bars=max_bars,
                             use_macd=use_macd, use_rsi=use_rsi, use_kd=use_kd)
        if not div["signals"]:
            return None

        # 進階過濾（預擷取 numpy）
        c_v   = closes.values
        ma5_v = ind["ma5"].values
        ma10_v= ind["ma10"].values
        n = len(c_v)
        c_y    = float(c_v[-2]) if n >= 2 else float(c_v[-1])
        ma5_y  = float(ma5_v[max(0, n-6):n-1].mean()) if n >= 6 else 0
        ma10_y = float(ma10_v[max(0, n-11):n-1].mean()) if n >= 11 else 0
        cond_ma = (ma5_y > 0 and c_y <= ma5_y * 1.02) or \
                  (ma10_y > 0 and c_y <= ma10_y * 1.02)
        strong = div["score"] >= 4 or div["regular_count"] >= 2
        if not (cond_ma or strong):
            return None

        close_now  = float(c_v[-1])
        prev_close = float(c_v[-2]) if n >= 2 else close_now
        change     = close_now - prev_close
        chg_pct    = (change / prev_close * 100) if prev_close > 0 else 0
        vol_ratio  = _last_float(ind["vol_ratio"].values, 1.0)
        rsi_val    = _last_float(ind["rsi"].values, 50.0)
        k_val      = _last_float(ind["K"].values, 50.0)
        dif_val    = _last_float(ind["dif"].values)
        dea_val    = _last_float(ind["dea"].values)

        name = next((s.get("name", "") for s in watchlist if s["id"] == sid), sid)

        return {
            "id": sid, "name": name, "ex": ex, "mode": mode,
            "close":      round(close_now, 2),
            "change":     round(change, 2),
            "change_pct": round(chg_pct, 2),
            "vol_ratio":  round(vol_ratio, 1),
            "rsi":        round(rsi_val, 1),
            "k_val":      round(k_val, 1),
            "dif":        round(dif_val, 4),
            "dea":        round(dea_val, 4),
            "macd_bull":  dif_val > dea_val,
            "div_score":  div["score"],
            "div_summary": div["summary"],
            "signals":    div["signals"],
            "regular_count": div["regular_count"],
            "hidden_count":  div["hidden_count"],
            "scanned_at": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return None


# ── 全市場掃描 ────────────────────────────────────────────
def run_scan(mode="daily", min_score=2, div_type="all"):
    global scan_progress
    stocks = list(watchlist)
    total  = len(stocks)
    scan_progress = {"done": 0, "total": total, "running": True, "mode": mode}
    label = {"daily": "日K", "weekly": "週K", "kd": "KD背離"}.get(mode, "日K")
    print(f"\n  🔍 {label}背離掃描：{total} 支，最低分數 {min_score}...")
    t0 = time.time()
    results = []

    futures = {_executor.submit(analyze_one, s, mode): s for s in stocks}
    for future in as_completed(futures):
        try:
            r = future.result(timeout=25)
            if r and r["div_score"] >= min_score:
                if div_type == "regular" and r["regular_count"] == 0: pass
                elif div_type == "hidden"  and r["hidden_count"]  == 0: pass
                else:
                    results.append(r)
        except Exception:
            pass
        scan_progress["done"] += 1

    elapsed = round(time.time() - t0, 1)
    scan_progress["running"] = False
    results.sort(key=lambda x: (-x["div_score"], -x.get("change_pct", 0)))
    print(f"  ✅ {label}背離完成：{len(results)} 支，{elapsed} 秒")
    return results, elapsed


# ── 日週雙重背離 ──────────────────────────────────────────
def analyze_daily_weekly(stock):
    sid = stock["id"]; ex = stock.get("ex", "tse")
    try:
        df = get_hist(sid, ex, period="1y", interval="1d")
        if len(df) < 60:
            return None

        closes  = df["Close"]
        highs   = df["High"]   if "High"   in df.columns else closes
        lows    = df["Low"]    if "Low"    in df.columns else closes
        volumes = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=closes.index)
        opens   = df["Open"]   if "Open"   in df.columns else closes

        ind_d = calc_ind(closes, highs, lows, volumes)
        div_d = calc_div_score(ind_d, closes, highs, lows, volumes, max_bars=15)
        if div_d["score"] < 5:
            return None

        # ★ 用快取的 weekly resample
        wdf = _get_weekly(sid, df)

        weekly_score = 0; weekly_sigs = []
        weekly_macd_bull = False; weekly_kd_cross = False; weekly_rsi_ok = False

        if len(wdf) >= 12:
            wc = wdf["Close"]; wh = wdf["High"]; wl = wdf["Low"]; wv = wdf["Volume"]
            ind_w = calc_ind(wc, wh, wl, wv)
            div_w = calc_div_score(ind_w, wc, wh, wl, wv, max_bars=10)
            weekly_score = div_w["score"]
            weekly_sigs  = div_w["signals"]

            wdif_v = ind_w["dif"].values
            wdea_v = ind_w["dea"].values
            weekly_macd_bull = float(wdif_v[-1]) > float(wdea_v[-1])

            wK_v = ind_w["K"].values; wD_v = ind_w["D"].values
            wk0, wd0 = float(wK_v[-1]), float(wD_v[-1])
            wk1, wd1 = float(wK_v[-2]), float(wD_v[-2])
            weekly_kd_cross  = wk0 > wd0 and wk1 <= wd1

            wrsi = _last_float(ind_w["rsi"].values, 50.0)
            weekly_rsi_ok = wrsi > 40

        c_v = closes.values; o_v = opens.values
        n   = len(c_v)
        c0  = float(c_v[-1])
        c1  = float(c_v[-2]) if n >= 2 else c0
        o0  = float(o_v[-1])
        chg = c0 - c1
        chg_pct = (chg / c1 * 100) if c1 > 0 else 0
        today_red = c0 > o0

        vr = _last_float(ind_d["vol_ratio"].values, 1.0)
        ma5_v  = ind_d["ma5"].values
        ma10_v = ind_d["ma10"].values
        ma5    = float(ma5_v[-1])
        ma10   = float(ma10_v[-1])
        ma5_rising = (float(ma5_v[-1]) > float(ma5_v[-2])) if n >= 2 else False
        above_ma5  = c0 > ma5
        above_ma10 = c0 > ma10

        dif_v = ind_d["dif"].values; dea_v = ind_d["dea"].values
        macd_bull = float(dif_v[-1]) > float(dea_v[-1])

        name = next((s.get("name", "") for s in watchlist if s["id"] == sid), sid)

        w_state = []
        if weekly_macd_bull: w_state.append("週MACD多")
        if weekly_kd_cross:  w_state.append("週KD黃金交叉")
        if weekly_rsi_ok:    w_state.append("週RSI>40")
        if not w_state:      w_state.append("週線待確認")

        is_best = today_red and above_ma5 and ma5_rising

        return {
            "id": sid, "name": name, "ex": ex,
            "close":      round(c0, 2),
            "change":     round(chg, 2),
            "change_pct": round(chg_pct, 2),
            "today_red":  today_red,
            "vol_ratio":  round(vr, 1),
            "above_ma5":  above_ma5,
            "above_ma10": above_ma10,
            "ma5_rising": ma5_rising,
            "macd_bull":  macd_bull,
            "div_score":  div_d["score"],
            "signals":    div_d["signals"],
            "div_summary": div_d["summary"],
            "regular_count": div_d["regular_count"],
            "hidden_count":  div_d["hidden_count"],
            "weekly_score":  weekly_score,
            "weekly_signals": weekly_sigs,
            "weekly_macd_bull": weekly_macd_bull,
            "weekly_kd_cross":  weekly_kd_cross,
            "weekly_rsi_ok":    weekly_rsi_ok,
            "weekly_state":     "、".join(w_state),
            "double_resonance": div_d["score"] >= 5 and weekly_score >= 2,
            "strong_resonance": div_d["score"] >= 5 and weekly_score >= 4,
            "is_best":   is_best,
            "scanned_at": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return None


def run_dw_scan(mode="triple", min_weekly=2):
    global scan_progress
    stocks = list(watchlist); total = len(stocks)
    label  = "日週強烈共振" if mode == "resonance" else "三指標全部背離+日週雙重"

    # ★ 關鍵修正：dw 掃描需要 1y 歷史，先批次補齊（一次性、之後 hit 快取）
    ensure_1y_cached()

    scan_progress = {"done": 0, "total": total, "running": True, "mode": mode}
    print(f"\n  🔍 {label}掃描：{total} 支...")
    t0 = time.time(); results = []

    futures = {_executor.submit(analyze_daily_weekly, s): s for s in stocks}
    for future in as_completed(futures):
        try:
            r = future.result(timeout=35)
            if r:
                if mode == "resonance" and not r["strong_resonance"]: pass
                elif mode == "triple"  and not r["double_resonance"]: pass
                else:
                    results.append(r)
        except Exception:
            pass
        scan_progress["done"] += 1

    elapsed = round(time.time() - t0, 1)
    scan_progress["running"] = False
    results.sort(key=lambda x: (
        -int(x["is_best"]),
        -x["weekly_score"],
        -x["div_score"],
        -x["change_pct"],
    ))
    print(f"  ✅ {label}完成：{len(results)} 支，{elapsed} 秒")
    return results, elapsed


# ── API ───────────────────────────────────────────────────
@app.route("/api/scan", methods=["GET"])
def api_scan():
    mode      = request.args.get("mode", "daily")
    min_score = int(request.args.get("min_score", 2))
    div_type  = request.args.get("type", "all")
    try:
        results, elapsed = run_scan(mode=mode, min_score=min_score, div_type=div_type)
        reg_cnt = sum(1 for r in results if r["regular_count"] > 0)
        hid_cnt = sum(1 for r in results if r["hidden_count"]  > 0)
        multi   = sum(1 for r in results if r["div_score"] >= 4)
        return jsonify(sanitize({
            "data":           results,
            "total_scanned":  len(watchlist),
            "hit_count":      len(results),
            "regular_count":  reg_cnt,
            "hidden_count":   hid_cnt,
            "multi_count":    multi,
            "mode":           mode,
            "elapsed":        elapsed,
            "scanned_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "data": []}), 500


@app.route("/api/scan/dw", methods=["GET"])
def api_scan_dw():
    mode = request.args.get("mode", "triple")
    try:
        results, elapsed = run_dw_scan(mode=mode)
        best   = sum(1 for r in results if r["is_best"])
        strong = sum(1 for r in results if r["strong_resonance"])
        w_macd = sum(1 for r in results if r["weekly_macd_bull"])
        w_kd   = sum(1 for r in results if r["weekly_kd_cross"])
        return jsonify(sanitize({
            "data":             results,
            "total_scanned":    len(watchlist),
            "hit_count":        len(results),
            "best_count":       best,
            "strong_count":     strong,
            "weekly_macd_bull": w_macd,
            "weekly_kd_cross":  w_kd,
            "mode":             mode,
            "elapsed":          elapsed,
            "scanned_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "data": []}), 500


@app.route("/api/progress", methods=["GET"])
def api_progress():
    return jsonify(scan_progress)


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({"watchlist": len(watchlist), "hist_cached": len(hist_cache),
                    "preload_done": preload_done,
                    "time": datetime.now().strftime("%H:%M:%S")})


# ══════════════════════════════════════════════════════════
# ▌ 條件 A/B/C — 7天累積表格（後端：掃描 + 儲存 + 刪除）
#   A = (三指標背離) 或 (日週雙重背離) 或 (日週共振)        ← OR
#   B = 週威廉波浪完成 + 日週MACD共振（日MACD多頭 且 週MACD多頭）
#   C = 週威廉波浪完成 + 日威廉波浪完成
#   每個條件各一張表：掃到的股票累積保留 7 天，超過 7 天自動移除，可手動刪除。
# ══════════════════════════════════════════════════════════
COND_TABLE_FILE = "cond_tables.json"
COND_KEEP_DAYS  = 7
cond_tables = {"A": {}, "B": {}, "C": {}}   # {cond: {sid: record}}
cond_lock   = threading.Lock()


def _load_cond_tables():
    global cond_tables
    try:
        if os.path.exists(COND_TABLE_FILE):
            with open(COND_TABLE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            for k in ("A", "B", "C"):
                if isinstance(data.get(k), dict):
                    cond_tables[k] = data[k]
    except Exception:
        pass


def _save_cond_tables():
    try:
        with open(COND_TABLE_FILE, "w", encoding="utf-8") as f:
            json.dump(cond_tables, f, ensure_ascii=False)
    except Exception:
        pass


def _expire_old(cond):
    """移除 last_seen 距今超過 7 天的紀錄（含今天往前共 7 個日曆日）。"""
    cutoff = (datetime.now() - timedelta(days=COND_KEEP_DAYS - 1)).strftime("%Y-%m-%d")
    tbl = cond_tables.get(cond, {})
    for sid in [s for s, r in tbl.items() if r.get("last_seen", "") < cutoff]:
        tbl.pop(sid, None)


def _merge_hits(cond, hits):
    """併入本次掃到的 hits：新股記 first_seen，舊股更新 last_seen 與最新數據。"""
    today = datetime.now().strftime("%Y-%m-%d")
    tbl = cond_tables.setdefault(cond, {})
    for h in hits:
        sid = h.get("id")
        if not sid:
            continue
        old = tbl.get(sid)
        rec = dict(h)
        rec["first_seen"] = old.get("first_seen", today) if old else today
        rec["last_seen"]  = today
        tbl[sid] = rec


def _cond_listing(cond):
    """先過期、再回傳排序後清單（新到舊 → 漲幅）。呼叫端需持有 cond_lock。"""
    _expire_old(cond)
    rows = list(cond_tables.get(cond, {}).values())
    rows.sort(key=lambda r: (r.get("last_seen", ""), r.get("change_pct", 0)), reverse=True)
    return rows


def _wave_complete(closes, highs, lows):
    """威廉波浪完成判定（日線傳日OHLC、週線傳週OHLC）。回傳 (是否完成, 詳細)。"""
    try:
        res = calc_williams_signals_v4(closes, highs, lows)
        return bool(res.get("buy_signals")), res
    except Exception:
        return False, {}


def _base_record(sid, df, ind_d):
    closes = df["Close"]; c_v = closes.values; n = len(c_v)
    c0 = float(c_v[-1]); c1 = float(c_v[-2]) if n >= 2 else c0
    chg = c0 - c1; chg_pct = (chg / c1 * 100) if c1 > 0 else 0.0
    vr = _last_float(ind_d["vol_ratio"].values, 1.0)
    name = next((s.get("name", "") for s in watchlist if s["id"] == sid), sid)
    return {"id": sid, "name": name,
            "close": round(c0, 2), "change": round(chg, 2),
            "change_pct": round(chg_pct, 2), "vol_ratio": round(vr, 1),
            "scanned_at": datetime.now().strftime("%H:%M:%S")}


def analyze_A(stock):
    """A = 三指標背離 OR 日週雙重背離 OR 日週共振（任一即入表）。"""
    sid = stock["id"]; ex = stock.get("ex", "tse")
    try:
        df = get_hist(sid, ex, period="1y", interval="1d")
        if len(df) < 60:
            return None
        closes  = df["Close"]
        highs   = df["High"]   if "High"   in df.columns else closes
        lows    = df["Low"]    if "Low"    in df.columns else closes
        volumes = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=closes.index)
        ind_d = calc_ind(closes, highs, lows, volumes)
        div_d = calc_div_score(ind_d, closes, highs, lows, volumes, max_bars=15)
        inds  = {s["indicator"] for s in div_d["signals"]}
        triple = {"MACD", "RSI", "KD"}.issubset(inds)

        weekly_score = 0
        wdf = _get_weekly(sid, df)
        if len(wdf) >= 12:
            wc = wdf["Close"]; wh = wdf["High"]; wl = wdf["Low"]; wv = wdf["Volume"]
            ind_w = calc_ind(wc, wh, wl, wv)
            weekly_score = calc_div_score(ind_w, wc, wh, wl, wv, max_bars=10)["score"]
        double = div_d["score"] >= 5 and weekly_score >= 2
        reson  = div_d["score"] >= 5 and weekly_score >= 4
        if not (triple or double or reson):
            return None

        tags = []
        if triple: tags.append("三指標背離")
        if double: tags.append("日週雙重背離")
        if reson:  tags.append("日週共振")
        rec = _base_record(sid, df, ind_d)
        rec.update({"div_score": div_d["score"], "weekly_score": weekly_score,
                    "div_summary": div_d["summary"], "tags": tags})
        return rec
    except Exception:
        return None


def analyze_B(stock):
    """B = 週威廉波浪完成 + 日週MACD共振（日MACD多頭 且 週MACD多頭）。"""
    sid = stock["id"]; ex = stock.get("ex", "tse")
    try:
        df = get_hist(sid, ex, period="1y", interval="1d")
        if len(df) < 60:
            return None
        closes  = df["Close"]
        highs   = df["High"]   if "High"   in df.columns else closes
        lows    = df["Low"]    if "Low"    in df.columns else closes
        volumes = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=closes.index)
        ind_d = calc_ind(closes, highs, lows, volumes)
        dif_v = ind_d["dif"].values; dea_v = ind_d["dea"].values
        daily_macd_bull = float(dif_v[-1]) > float(dea_v[-1])

        wdf = _get_weekly(sid, df)
        if len(wdf) < 30:
            return None
        wc = wdf["Close"]; wh = wdf["High"]; wl = wdf["Low"]; wv = wdf["Volume"]
        ind_w = calc_ind(wc, wh, wl, wv)
        wdif = ind_w["dif"].values; wdea = ind_w["dea"].values
        weekly_macd_bull = float(wdif[-1]) > float(wdea[-1])
        if not (daily_macd_bull and weekly_macd_bull):
            return None

        wave_ok, wres = _wave_complete(wc, wh, wl)
        if not wave_ok:
            return None

        rec = _base_record(sid, df, ind_d)
        wi = wres.get("wave_info", {})
        rec.update({"tags": ["週威廉波浪", "日週MACD共振"],
                    "weekly_wr": wres.get("wr_now", 0),
                    "deep_low_val": wi.get("deep_low_val", 0),
                    "peak_count": wi.get("peak_count", 0),
                    "daily_macd_bull": daily_macd_bull,
                    "weekly_macd_bull": weekly_macd_bull})
        return rec
    except Exception:
        return None


def analyze_C(stock):
    """C = 週威廉波浪完成 + 日威廉波浪完成。"""
    sid = stock["id"]; ex = stock.get("ex", "tse")
    try:
        df = get_hist(sid, ex, period="1y", interval="1d")
        if len(df) < 60:
            return None
        closes  = df["Close"]
        highs   = df["High"]   if "High"   in df.columns else closes
        lows    = df["Low"]    if "Low"    in df.columns else closes
        volumes = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=closes.index)
        ind_d = calc_ind(closes, highs, lows, volumes)

        d_ok, dres = _wave_complete(closes, highs, lows)
        if not d_ok:
            return None
        wdf = _get_weekly(sid, df)
        if len(wdf) < 30:
            return None
        w_ok, wres = _wave_complete(wdf["Close"], wdf["High"], wdf["Low"])
        if not w_ok:
            return None

        rec = _base_record(sid, df, ind_d)
        rec.update({"tags": ["週威廉波浪", "日威廉波浪"],
                    "daily_wr": dres.get("wr_now", 0),
                    "weekly_wr": wres.get("wr_now", 0)})
        return rec
    except Exception:
        return None


_COND_FN = {"A": analyze_A, "B": analyze_B, "C": analyze_C}


def run_cond_scan(cond):
    """掃描全市場 → 併入該條件表格 → 過期 → 回傳 (清單, 今日命中數, 耗時)。"""
    global scan_progress
    fn = _COND_FN[cond]
    ensure_1y_cached()
    stocks = list(watchlist); total = len(stocks)
    scan_progress = {"done": 0, "total": total, "running": True, "mode": f"cond{cond}"}
    print(f"\n  🔍 條件 {cond} 掃描：{total} 支...")
    t0 = time.time(); hits = []
    futures = {_executor.submit(fn, s): s for s in stocks}
    for fut in as_completed(futures):
        try:
            r = fut.result(timeout=35)
            if r:
                hits.append(r)
        except Exception:
            pass
        scan_progress["done"] += 1
    scan_progress["running"] = False
    elapsed = round(time.time() - t0, 1)
    with cond_lock:
        _merge_hits(cond, hits)
        rows = _cond_listing(cond)
        _save_cond_tables()
    print(f"  ✅ 條件 {cond} 完成：今日 {len(hits)} 支、表內共 {len(rows)} 支，{elapsed} 秒")
    return rows, len(hits), elapsed


@app.route("/api/cond/<cond>/scan", methods=["GET"])
def api_cond_scan(cond):
    cond = cond.upper()
    if cond not in _COND_FN:
        return jsonify({"error": "未知條件，僅支援 A/B/C", "data": []}), 400
    try:
        rows, today_n, elapsed = run_cond_scan(cond)
        return jsonify(sanitize({
            "data": rows, "cond": cond,
            "today_hits": today_n, "table_count": len(rows),
            "total_scanned": len(watchlist), "elapsed": elapsed,
            "keep_days": COND_KEEP_DAYS,
            "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "data": []}), 500


@app.route("/api/cond/<cond>", methods=["GET"])
def api_cond_list(cond):
    cond = cond.upper()
    if cond not in _COND_FN:
        return jsonify({"error": "未知條件，僅支援 A/B/C", "data": []}), 400
    with cond_lock:
        rows = _cond_listing(cond)
        _save_cond_tables()
    return jsonify(sanitize({"data": rows, "cond": cond,
                             "table_count": len(rows), "keep_days": COND_KEEP_DAYS}))


@app.route("/api/cond/<cond>/delete", methods=["POST"])
def api_cond_delete(cond):
    cond = cond.upper()
    if cond not in _COND_FN:
        return jsonify({"error": "未知條件，僅支援 A/B/C"}), 400
    data = request.get_json(silent=True) or {}
    sid = str(data.get("id", "")).strip()
    if not sid:
        return jsonify({"error": "缺少股號 id"}), 400
    with cond_lock:
        existed = cond_tables.get(cond, {}).pop(sid, None) is not None
        rows = _cond_listing(cond)
        _save_cond_tables()
    return jsonify(sanitize({"ok": existed, "removed": sid, "cond": cond,
                             "data": rows, "table_count": len(rows)}))


# ══════════════════════════════════════════════════════════
# ▌ 前端 HTML — XQ 風格大字體表格
# ══════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>指標背離選股系統</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#000;color:#e0e0e0;font-family:'Microsoft JhengHei','微軟正黑體','Noto Sans TC',sans-serif;font-size:14px}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#111}
::-webkit-scrollbar-thumb{background:#333;border-radius:3px}

.hdr{background:linear-gradient(90deg,#0a0a1a,#111);border-bottom:2px solid #1a3a5c;
  padding:10px 16px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100}
.brand-row{display:flex;align-items:center;gap:10px}
.brand-icon{width:36px;height:36px;border-radius:6px;background:rgba(206,147,216,.15);
  border:1px solid #CE93D8;display:flex;align-items:center;justify-content:center;font-size:18px}
.brand-name{font-size:16px;font-weight:700;color:#fff;letter-spacing:1px}
.brand-sub{font-size:11px;color:#4a6080;margin-top:1px}
.hdr-right{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.status-pill{font-size:11px;padding:3px 10px;border-radius:12px;
  border:1px solid rgba(255,255,255,.1);color:#5a7090;font-family:monospace}
.status-pill.live{border-color:#4CAF50;color:#4CAF50}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.blink{animation:blink 1.4s infinite}

.scan-bar{padding:10px 16px;background:#0d0d1a;border-bottom:1px solid #1a2a3a;
  display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn-day,.btn-week,.btn-kd,.btn-triple,.btn-resonance{
  padding:10px 22px;font-size:15px;font-weight:700;
  border:none;border-radius:6px;cursor:pointer;transition:all .2s;
  color:#fff;letter-spacing:1px}
.btn-day{background:linear-gradient(135deg,#1565C0,#1976D2);box-shadow:0 2px 12px rgba(21,101,192,.4)}
.btn-week{background:linear-gradient(135deg,#6A1B9A,#8E24AA);box-shadow:0 2px 12px rgba(106,27,154,.4)}
.btn-kd{background:linear-gradient(135deg,#00695C,#00897B);box-shadow:0 2px 12px rgba(0,105,92,.4)}
.btn-triple{background:linear-gradient(135deg,#B71C1C,#E53935);box-shadow:0 2px 12px rgba(183,28,28,.4)}
.btn-resonance{background:linear-gradient(135deg,#4A148C,#7B1FA2);box-shadow:0 2px 12px rgba(74,20,140,.4)}
.btn-day:hover,.btn-week:hover,.btn-kd:hover,.btn-triple:hover,.btn-resonance:hover{transform:translateY(-1px)}
.btn-day:disabled,.btn-week:disabled,.btn-kd:disabled,.btn-triple:disabled,.btn-resonance:disabled{opacity:.4;cursor:not-allowed;transform:none}

.btn-condA,.btn-condB,.btn-condC{padding:10px 22px;font-size:15px;font-weight:700;
  border:none;border-radius:6px;cursor:pointer;transition:all .2s;color:#fff;letter-spacing:1px}
.btn-condA{background:linear-gradient(135deg,#E65100,#FB8C00);box-shadow:0 2px 12px rgba(230,81,0,.4)}
.btn-condB{background:linear-gradient(135deg,#0D47A1,#1E88E5);box-shadow:0 2px 12px rgba(13,71,161,.4)}
.btn-condC{background:linear-gradient(135deg,#1B5E20,#43A047);box-shadow:0 2px 12px rgba(27,94,32,.4)}
.btn-condA:hover,.btn-condB:hover,.btn-condC:hover{transform:translateY(-1px)}
.btn-condA:disabled,.btn-condB:disabled,.btn-condC:disabled{opacity:.4;cursor:not-allowed;transform:none}

.cond-tbl{width:100%;border-collapse:collapse}
.cond-tbl th{position:sticky;top:57px;background:#0a1428;color:#5a8ab0;font-size:12px;
  text-align:left;padding:10px 12px;border-bottom:2px solid #1a3a5c;white-space:nowrap;z-index:5}
.cond-tbl td{padding:12px;border-bottom:1px solid #0f1520;white-space:nowrap;vertical-align:middle}
.cond-tbl tr:hover td{background:#0a1428}
.cond-tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:3px;margin:1px;
  background:rgba(206,147,216,.18);color:#E1BEE7;white-space:nowrap}
.del-btn{padding:5px 12px;font-size:12px;font-weight:600;border:1px solid rgba(239,83,80,.5);
  border-radius:5px;background:transparent;color:#EF9A9A;cursor:pointer;transition:all .15s}
.del-btn:hover{background:rgba(239,83,80,.2);border-color:#EF5350;color:#fff}

.btn-export{padding:8px 16px;font-size:13px;background:transparent;
  border:1px solid rgba(255,255,255,.15);color:#7c8fa8;border-radius:5px;cursor:pointer}
.btn-export:hover{border-color:#a5d6a7;color:#a5d6a7}

.filter-row{display:flex;gap:8px;align-items:center}
.filter-row select{background:#0d1520;color:#9bb3cc;border:1px solid rgba(255,255,255,.12);
  border-radius:5px;padding:7px 10px;font-size:12px;outline:none;cursor:pointer;
  font-family:monospace}
.scan-info{font-size:12px;color:#4a6080;font-family:monospace;padding-left:4px}

.mode-bar{padding:6px 16px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid #111;min-height:34px}
.mode-tag{font-size:13px;font-weight:700;padding:3px 14px;border-radius:4px;letter-spacing:1px}
.mode-day{background:rgba(21,101,192,.2);color:#64B5F6;border:1px solid #1565C0}
.mode-week{background:rgba(106,27,154,.2);color:#CE93D8;border:1px solid #7B1FA2}
.mode-kd{background:rgba(0,105,92,.2);color:#80CBC4;border:1px solid #00695C}
.mode-triple{background:rgba(183,28,28,.2);color:#EF9A9A;border:1px solid #B71C1C}
.mode-resonance{background:rgba(74,20,140,.2);color:#CE93D8;border:1px solid #4A148C}
.mode-none{color:#3d5060;font-size:12px}

.prog-wrap{height:3px;background:#111;overflow:hidden}
.prog-fill{height:100%;background:linear-gradient(90deg,#1565C0,#CE93D8);
  transition:width .4s ease;border-radius:2px}

.stats-bar{padding:8px 16px;background:#050510;border-bottom:1px solid #111;
  display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.stat{text-align:center}
.stat-n{font-family:monospace;font-size:18px;font-weight:700}
.stat-l{font-size:10px;color:#3d5060;text-transform:uppercase;letter-spacing:.05em;margin-top:1px}

.tbl-wrap{overflow-x:auto}
.tbl-head{display:grid;
  grid-template-columns: 60px 1fr 80px 130px 130px 120px 70px 180px 160px 80px;
  background:#0a1428;border-bottom:2px solid #1a3a5c;
  padding:0;position:sticky;top:57px;z-index:5}
.th{padding:10px 12px;font-size:12px;font-weight:700;color:#5a8ab0;
  text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;
  border-right:1px solid #0f1e2e}
.th.r{text-align:right}
.th:last-child{border-right:none}

.tbl-body{}
.row{display:grid;
  grid-template-columns: 60px 1fr 80px 130px 130px 120px 70px 180px 160px 80px;
  border-bottom:1px solid #0f1520;transition:background .1s}
.row:hover{background:#0a1428}
.cell{padding:12px 12px;display:flex;align-items:center;
  border-right:1px solid #0a0f18;white-space:nowrap;overflow:hidden}
.cell:last-child{border-right:none}
.cell.r{justify-content:flex-end}

.stock-id{font-size:20px;font-weight:700;color:#fff;letter-spacing:1px;line-height:1}

.price{font-size:24px;font-weight:700;color:#fff;font-family:monospace;letter-spacing:1px}
.chg-up,.pct-up{font-size:22px;font-weight:700;color:#FF3B3B;font-family:monospace}
.chg-dn,.pct-dn{font-size:22px;font-weight:700;color:#00C853;font-family:monospace}
.chg-zero,.pct-zero{font-size:22px;font-weight:700;color:#888;font-family:monospace}

.score-badge{width:36px;height:36px;border-radius:8px;
  display:inline-flex;align-items:center;justify-content:center;
  font-family:monospace;font-weight:700;font-size:16px}
.sc7,.sc6{background:rgba(206,147,216,.2);color:#CE93D8;border:1px solid rgba(206,147,216,.4)}
.sc5,.sc4{background:rgba(165,214,167,.2);color:#81C784;border:1px solid rgba(165,214,167,.4)}
.sc3,.sc2{background:rgba(100,181,246,.2);color:#64B5F6;border:1px solid rgba(100,181,246,.4)}
.sc1,.sc0{background:rgba(255,255,255,.05);color:#fff;border:1px solid rgba(255,255,255,.15)}

.sig-tags{display:flex;gap:3px;flex-wrap:wrap}
.sig-r{font-size:11px;padding:2px 7px;border-radius:3px;
  background:rgba(206,147,216,.15);color:#fff;white-space:nowrap}
.sig-h{font-size:11px;padding:2px 7px;border-radius:3px;
  background:rgba(128,222,234,.12);color:#fff;white-space:nowrap}

.macd-up{color:#FF5252;font-size:13px}
.macd-dn{color:#00E676;font-size:13px}
.time-val{font-family:monospace;font-size:11px;color:#fff}

.empty{text-align:center;padding:80px 20px;color:#2d3748}
.empty-icon{font-size:60px;margin-bottom:20px;display:block}
.empty-title{font-size:16px;color:#3d5060;margin-bottom:8px}
.empty-sub{font-size:13px;color:#2d3748;line-height:1.7}

.vbtn{padding:5px 13px;font-size:12px;font-weight:600;
  border:1px solid rgba(255,255,255,.15);border-radius:5px;
  background:transparent;color:#7c8fa8;cursor:pointer;
  transition:all .15s;white-space:nowrap;letter-spacing:.3px}
.vbtn:hover{border-color:#fff;color:#fff}
.vbtn.active{background:rgba(255,255,255,.12);border-color:#fff;color:#fff}
.vbtn.red{border-color:rgba(239,83,80,.4);color:#EF9A9A}
.vbtn.red.active{background:rgba(239,83,80,.15);border-color:#EF5350;color:#fff}
.vbtn.green{border-color:rgba(129,199,132,.4);color:#81C784}
.vbtn.green.active{background:rgba(129,199,132,.15);border-color:#81C784;color:#fff}
.vbtn.purple{border-color:rgba(206,147,216,.4);color:#CE93D8}
.vbtn.purple.active{background:rgba(206,147,216,.15);border-color:#CE93D8;color:#fff}
.vbtn.amber{border-color:rgba(255,183,77,.4);color:#FFB74D}
.vbtn.amber.active{background:rgba(255,183,77,.15);border-color:#FFB74D;color:#fff}

.legend{padding:8px 16px;background:#050510;border-top:1px solid #0f1520;
  display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#3d5060}
.leg-item{display:flex;align-items:center;gap:5px}
.leg-r{width:10px;height:10px;border-radius:2px;background:#CE93D8;flex-shrink:0}
.leg-h{width:10px;height:10px;border-radius:2px;background:#80DEEA;flex-shrink:0}
</style>
</head>
<body>

<div class="hdr">
  <div class="brand-row">
    <div class="brand-icon">🔮</div>
    <div>
      <div class="brand-name">指標背離選股系統</div>
      <div class="brand-sub">MACD · RSI · KD 底背離偵測 · 極速版</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="status-pill live"><span class="blink">●</span> 即時掃描</span>
    <span class="status-pill" id="time-badge">--:--:--</span>
    <span class="status-pill" id="total-badge">載入中...</span>
  </div>
</div>

<div class="scan-bar">
  <div class="filter-row">
    <button class="btn-condA" id="btn-condA" onclick="startCondScan('A')">🟠 條件A</button>
    <button class="btn-condB" id="btn-condB" onclick="startCondScan('B')">🔵 條件B</button>
    <button class="btn-condC" id="btn-condC" onclick="startCondScan('C')">🟢 條件C</button>
    <button class="btn-triple"    id="btn-triple"    onclick="startDWScan('triple')">🔴 三指標+日週雙重</button>
    <button class="btn-resonance" id="btn-resonance" onclick="startDWScan('resonance')">🔮 日週強烈共振</button>
    <button class="btn-export" onclick="exportCSV()">📥 匯出CSV</button>
  </div>
  <span class="scan-info" id="scan-info"></span>
</div>

<div class="prog-wrap"><div class="prog-fill" id="prog" style="width:0%"></div></div>

<div class="mode-bar" id="mode-bar">
  <span class="mode-none">請點擊掃描按鈕開始選股</span>
</div>

<div class="stats-bar" id="stats-bar" style="display:none">
  <div class="stat"><div class="stat-n" id="s-hit" style="color:#81C784">0</div><div class="stat-l">背離股票</div></div>
  <div class="stat"><div class="stat-n" id="s-reg" style="color:#CE93D8">0</div><div class="stat-l">正規底背離</div></div>
  <div class="stat"><div class="stat-n" id="s-hid" style="color:#80DEEA">0</div><div class="stat-l">隱藏底背離</div></div>
  <div class="stat"><div class="stat-n" id="s-mul" style="color:#FFD54F">0</div><div class="stat-l">多指標共振</div></div>
  <div class="stat"><div class="stat-n" id="s-scn" style="color:#4a6080">0</div><div class="stat-l">掃描總數</div></div>
  <div class="stat"><div class="stat-n" id="s-time" style="color:#4a6080">--</div><div class="stat-l">耗時(秒)</div></div>
</div>

<div id="view-bar" style="display:none;padding:8px 16px;background:#07090e;
  border-bottom:1px solid #0f1525;flex-wrap:wrap;gap:6px;align-items:center">
  <span style="font-size:11px;color:#4a6080;font-family:monospace;margin-right:4px">📋 快速檢視：</span>
  <button class="vbtn active" id="vb-all"     onclick="viewFilter('all')">全部結果</button>
  <button class="vbtn"        id="vb-regular" onclick="viewFilter('regular')">正規底背離</button>
  <button class="vbtn"        id="vb-hidden"  onclick="viewFilter('hidden')">隱藏底背離</button>
  <button class="vbtn red"    id="vb-s7"      onclick="viewFilter('s7')">分數≥7（最強）</button>
  <button class="vbtn red"    id="vb-s5"      onclick="viewFilter('s5')">分數≥5</button>
  <button class="vbtn green"  id="vb-up3"     onclick="viewFilter('up3')">今日漲≥3%</button>
  <button class="vbtn green"  id="vb-up"      onclick="viewFilter('up')">今日上漲</button>
  <button class="vbtn purple" id="vb-best"    onclick="viewFilter('best')">🎯 最佳買點</button>
  <button class="vbtn purple" id="vb-wmacd"   onclick="viewFilter('wmacd')">週MACD多頭</button>
  <button class="vbtn amber"  id="vb-wkd"     onclick="viewFilter('wkd')">週KD黃金交叉</button>
  <button class="vbtn amber"  id="vb-macd"    onclick="viewFilter('macd')">日MACD多頭</button>
  <span id="view-count" style="font-size:11px;color:#4a6080;font-family:monospace;margin-left:8px"></span>
</div>

<div class="tbl-head" id="tbl-head" style="display:none">
  <div class="th">#</div>
  <div class="th">股票</div>
  <div class="th r" style="line-height:1.4">日K分<br><span style="color:#CE93D8">週K分</span></div>
  <div class="th r">股　　價</div>
  <div class="th r">漲　　跌</div>
  <div class="th r">漲　幅%</div>
  <div class="th r">量比</div>
  <div class="th">背離訊號</div>
  <div class="th">週線狀態</div>
  <div class="th r">時間</div>
</div>

<div class="tbl-wrap">
  <div class="tbl-body" id="tbl-body">
    <div class="empty">
      <span class="empty-icon">🔮</span>
      <div class="empty-title">點擊掃描按鈕開始選股</div>
      <div class="empty-sub">
        <b style="color:#64B5F6">日K背離掃描</b>：偵測近15根內的指標背離，適合短線操作<br>
        <b style="color:#CE93D8">週K背離掃描</b>：偵測週線背離，訊號更強、持續性更高<br>
        MACD + RSI + KD 三指標同時偵測，多指標共振分數更高
      </div>
    </div>
  </div>
</div>

<div class="legend">
  <div class="leg-item"><div class="leg-r"></div>正規底背離：價格創新低，指標未創新低 → 潛在底部反轉</div>
  <div class="leg-item"><div class="leg-h"></div>隱藏底背離：價格低點更高，指標更低 → 上升趨勢延續</div>
  <div class="leg-item" style="color:#FFD54F">★ 分數4以上 = 多指標共振，高機率訊號</div>
</div>

<script>
let allData = [];
let scanMode = '';
let progTimer = null;
let scanStart = 0;

setInterval(() => {
  document.getElementById('time-badge').textContent =
    new Date().toLocaleTimeString('zh-TW');
}, 1000);

fetch('/api/status').then(r=>r.json()).then(d=>{
  document.getElementById('total-badge').textContent = d.watchlist + ' 支股票';
}).catch(()=>{});

function setBtns(disabled) {
  ['btn-day','btn-week','btn-kd','btn-triple','btn-resonance','btn-condA','btn-condB','btn-condC'].forEach(id => {
    const el = document.getElementById(id); if (el) el.disabled = disabled;
  });
}

function startScan(mode) {
  const type  = document.getElementById('sel-type').value;
  const score = document.getElementById('sel-score').value;
  scanMode = mode;
  setBtns(true);
  document.getElementById('stats-bar').style.display = 'none';
  document.getElementById('tbl-head').style.display  = 'none';
  document.getElementById('view-bar').style.display  = 'none';
  document.getElementById('scan-info').textContent = '';
  document.getElementById('prog').style.width = '0%';

  const label = mode === 'daily' ? '日K' : mode === 'weekly' ? '週K' : 'KD背離';
  const color = mode === 'daily' ? '#64B5F6' : '#CE93D8';
  document.getElementById('mode-bar').innerHTML =
    `<span class="mode-tag ${mode==='daily'?'mode-day':mode==='weekly'?'mode-week':'mode-kd'}">● ${label}背離掃描進行中...</span>`;
  document.getElementById('tbl-body').innerHTML =
    `<div class="empty"><span class="empty-icon" style="animation:blink 1s infinite">🔮</span>
     <div class="empty-title" style="color:${color}">正在掃描全台股${label}背離...</div>
     <div class="empty-sub" id="prog-text">0 / 0 (0%)</div></div>`;

  scanStart = Date.now();
  clearInterval(progTimer);
  progTimer = setInterval(() => {
    fetch('/api/progress').then(r=>r.json()).then(d=>{
      if (d.total > 0) {
        const pct = Math.round(d.done / d.total * 100);
        document.getElementById('prog').style.width = pct + '%';
        const pt = document.getElementById('prog-text');
        if (pt) pt.textContent = `${d.done} / ${d.total} (${pct}%)`;
        document.getElementById('scan-info').textContent = `${d.done}/${d.total}`;
        if (!d.running) clearInterval(progTimer);
      }
    }).catch(()=>{});
  }, 600);

  fetch(`/api/scan?mode=${mode}&type=${type}&min_score=${score}`)
    .then(r => r.json())
    .then(data => {
      clearInterval(progTimer);
      document.getElementById('prog').style.width = '100%';
      setTimeout(() => document.getElementById('prog').style.width = '0%', 1500);

      allData = data.data || [];
      const elapsed = data.elapsed || ((Date.now()-scanStart)/1000).toFixed(1);
      const label2 = mode === 'daily' ? '日K' : mode === 'weekly' ? '週K' : 'KD背離';

      document.getElementById('s-hit').textContent  = data.hit_count     || 0;
      document.getElementById('s-reg').textContent  = data.regular_count || 0;
      document.getElementById('s-hid').textContent  = data.hidden_count  || 0;
      document.getElementById('s-mul').textContent  = data.multi_count   || 0;
      document.getElementById('s-scn').textContent  = data.total_scanned || 0;
      document.getElementById('s-time').textContent = elapsed;
      document.getElementById('stats-bar').style.display = 'flex';

      document.getElementById('mode-bar').innerHTML =
        `<span class="mode-tag ${mode==='daily'?'mode-day':mode==='weekly'?'mode-week':'mode-kd'}">${label2}背離</span>
         <span style="color:#3d5060;font-size:12px">
           共掃描 ${data.total_scanned} 支 · 找到 ${data.hit_count} 支 · 耗時 ${elapsed} 秒
         </span>`;

      document.getElementById('scan-info').textContent =
        `✓ ${data.hit_count} 支 · ${elapsed}秒`;
      renderTable(allData);
      document.getElementById('tbl-head').style.display = 'grid';
      showViewBar();
      setBtns(false);
    })
    .catch(e => {
      clearInterval(progTimer);
      document.getElementById('tbl-body').innerHTML =
        `<div class="empty"><span class="empty-icon">⚠</span>
         <div style="color:#ef5350">掃描失敗：${e.message}</div></div>`;
      document.getElementById('mode-bar').innerHTML =
        '<span class="mode-none" style="color:#ef5350">掃描失敗，請重試</span>';
      setBtns(false);
    });
}

function sortResults() {
  const key = document.getElementById('sel-sort').value;
  const sorted = [...allData].sort((a,b) => (b[key]||0) - (a[key]||0));
  renderTable(sorted);
}

function scoreClass(sc) {
  if (sc >= 6) return 'sc6';
  if (sc >= 5) return 'sc5';
  if (sc >= 4) return 'sc4';
  if (sc >= 3) return 'sc3';
  return 'sc2';
}

function renderTable(data) {
  if (!data || data.length === 0) {
    document.getElementById('tbl-head').style.display = 'none';
    document.getElementById('tbl-body').innerHTML =
      `<div class="empty"><span class="empty-icon">🔇</span>
       <div class="empty-title">無符合條件的背離股票</div>
       <div class="empty-sub">請降低最低分數或切換背離類型</div></div>`;
    return;
  }
  document.getElementById('tbl-head').style.display = 'grid';
  const rows = data.map((r, i) => {
    const up    = r.change_pct > 0;
    const zero  = r.change_pct === 0;
    const chgCls= zero ? 'chg-zero' : (up ? 'chg-up' : 'chg-dn');
    const pctCls= zero ? 'pct-zero' : (up ? 'pct-up' : 'pct-dn');
    const chgStr= (up?'▲':zero?'─':'▼') + ' ' + Math.abs(r.change).toFixed(2);
    const pctStr= (up?'+':'') + r.change_pct.toFixed(2) + '%';
    const tags = r.signals.map(s =>
      `<span class="${s.type==='regular'?'sig-r':'sig-h'}">${s.id}</span>`
    ).join('');
    const macdDir = r.macd_bull
      ? '<span class="macd-up">▲ 多頭</span>'
      : '<span class="macd-dn">▼ 空頭</span>';
    return `<div class="row">
      <div class="cell" style="font-family:monospace;font-size:15px;color:#fff">${i+1}</div>
      <div class="cell" style="flex-direction:column;align-items:flex-start">
        <div class="stock-id">${r.id}　${r.name}</div>
      </div>
      <div class="cell r">
        <span class="score-badge ${scoreClass(r.div_score)}">${r.div_score}</span>
      </div>
      <div class="cell r"><span class="price">${r.close.toFixed(2)}</span></div>
      <div class="cell r"><span class="${chgCls}">${chgStr}</span></div>
      <div class="cell r"><span class="${pctCls}">${pctStr}</span></div>
      <div class="cell r">${macdDir}</div>
      <div class="cell"><div class="sig-tags">${tags || '<span style="color:#2d3748">─</span>'}</div></div>
      <div class="cell"></div>
      <div class="cell r"><span class="time-val">${r.scanned_at}</span></div>
    </div>`;
  }).join('');
  document.getElementById('tbl-body').innerHTML = rows;
}

function startDWScan(mode) {
  setBtns(true);
  const label = mode === 'triple' ? '三指標全部背離+日週雙重' : '日週強烈共振';
  const color = mode === 'triple' ? '#EF9A9A' : '#CE93D8';
  scanMode = mode;
  document.getElementById('stats-bar').style.display = 'none';
  document.getElementById('tbl-head').style.display = 'none';
  document.getElementById('view-bar').style.display = 'none';
  document.getElementById('scan-info').textContent = '';
  document.getElementById('prog').style.width = '0%';
  document.getElementById('mode-bar').innerHTML =
    `<span class="mode-tag mode-${mode}">● ${label}掃描中...</span>`;
  document.getElementById('tbl-body').innerHTML =
    `<div class="empty"><span class="empty-icon" style="animation:blink 1s infinite">🔮</span>
     <div style="font-size:15px;color:${color}">${label}掃描中...</div>
     <div style="font-size:12px;color:#3d5060;margin-top:8px" id="prog-txt">0 / 0 (0%)</div></div>`;

  clearInterval(progTimer);
  progTimer = setInterval(() => {
    fetch('/api/progress').then(r=>r.json()).then(d=>{
      if (d.total > 0) {
        const pct = Math.round(d.done / d.total * 100);
        document.getElementById('prog').style.width = pct + '%';
        const pt = document.getElementById('prog-txt');
        if (pt) pt.textContent = `${d.done} / ${d.total} (${pct}%)`;
        document.getElementById('scan-info').textContent = `${d.done}/${d.total}`;
        if (!d.running) clearInterval(progTimer);
      }
    }).catch(()=>{});
  }, 700);

  fetch(`/api/scan/dw?mode=${mode}`)
    .then(r => r.json())
    .then(data => {
      clearInterval(progTimer);
      document.getElementById('prog').style.width = '100%';
      setTimeout(() => document.getElementById('prog').style.width = '0%', 1500);
      allData = data.data || [];

      document.getElementById('s-hit').textContent  = data.hit_count || 0;
      document.getElementById('s-reg').textContent  = data.best_count || 0;
      document.getElementById('s-hid').textContent  = data.strong_count || 0;
      document.getElementById('s-mul').textContent  = (data.weekly_kd_cross||0) + '週KD✕';
      document.getElementById('s-scn').textContent  = data.total_scanned || 0;
      document.getElementById('s-time').textContent = data.elapsed || '--';
      document.getElementById('stats-bar').style.display = 'flex';

      const label2 = mode === 'triple' ? '三指標+日週雙重' : '日週強烈共振';
      document.getElementById('mode-bar').innerHTML =
        `<span class="mode-tag mode-${mode}">${label2}</span>
         <span style="color:#3d5060;font-size:12px">
           共掃描 ${data.total_scanned} 支 · 找到 ${data.hit_count} 支
           · 最佳買點 ${data.best_count} 支 · 耗時 ${data.elapsed} 秒
         </span>`;

      document.getElementById('scan-info').textContent =
        `✓ ${data.hit_count} 支 · ${data.elapsed}秒`;

      renderDWTable(allData);
      document.getElementById('tbl-head').style.display = 'grid';
      showViewBar();
      setBtns(false);
    })
    .catch(e => {
      clearInterval(progTimer);
      document.getElementById('tbl-body').innerHTML =
        `<div class="empty"><span class="empty-icon">⚠</span>
         <div style="color:#ef5350">掃描失敗：${e.message}</div></div>`;
      document.getElementById('mode-bar').innerHTML =
        '<span class="mode-none" style="color:#ef5350">掃描失敗</span>';
      setBtns(false);
    });
}

function renderDWTable(data) {
  if (!data || data.length === 0) {
    document.getElementById('tbl-head').style.display = 'none';
    document.getElementById('tbl-body').innerHTML =
      `<div class="empty"><span class="empty-icon">🔇</span>
       <div>無符合條件的股票，請放寬設定或使用三指標全部背離模式</div></div>`;
    return;
  }
  document.getElementById('tbl-head').style.display = 'grid';
  const rows = data.map((r, i) => {
    const up    = r.change_pct > 0;
    const zero  = r.change_pct === 0;
    const cls   = zero ? 'chg-zero' : (up ? 'chg-up' : 'chg-dn');
    const arrow = up ? '▲ ' : (zero ? '─ ' : '▼ ');
    const chgStr= arrow + Math.abs(r.change).toFixed(2);
    const pctStr= (up ? '+' : '') + r.change_pct.toFixed(2) + '%';
    const dtags = (r.signals||[]).map(s =>
      `<span class="${s.type==='regular'?'sig-r':'sig-h'}">${s.id}</span>`
    ).join('');
    const wtags = (r.weekly_signals||[]).map(s =>
      `<span style="font-size:10px;padding:2px 6px;border-radius:3px;
        background:rgba(206,147,216,.2);color:#CE93D8;white-space:nowrap">週${s.id||s}</span>`
    ).join('');
    const wstate = r.weekly_state || '─';
    const bestTag = r.is_best
      ? '<span style="font-size:10px;padding:2px 8px;border-radius:3px;background:rgba(255,23,68,.2);color:#FF6B6B;font-weight:700;border:1px solid rgba(255,23,68,.3)">🎯 最佳買點</span>'
      : '';
    const redTag = r.today_red
      ? '<span style="color:#FF3B3B;font-size:12px">●紅</span>'
      : '<span style="color:#00C853;font-size:12px">●黑</span>';
    const maTag = r.ma5_rising
      ? '<span style="color:#FF9800;font-size:11px">MA5↑</span>'
      : '';
    return `<div class="row">
      <div class="cell" style="color:#4a6080;font-family:monospace;font-size:14px">${i+1}</div>
      <div class="cell" style="flex-direction:column;align-items:flex-start;gap:4px">
        <div class="stock-id">${r.id}　${r.name}</div>
        <div style="display:flex;gap:4px;flex-wrap:wrap">${bestTag} ${redTag} ${maTag}</div>
      </div>
      <div class="cell r" style="flex-direction:column;align-items:flex-end;gap:2px">
        <span style="font-family:monospace;font-size:18px;font-weight:700;color:#EF9A9A">${r.div_score}</span>
        <span style="font-family:monospace;font-size:18px;font-weight:700;color:#CE93D8">${r.weekly_score}</span>
      </div>
      <div class="cell r"><span class="price">${r.close.toFixed(2)}</span></div>
      <div class="cell r"><span class="${cls}">${chgStr}</span></div>
      <div class="cell r"><span class="${cls}" style="font-size:20px">${pctStr}</span></div>
      <div class="cell r" style="font-family:monospace;font-size:14px;color:#fff">${r.vol_ratio}x</div>
      <div class="cell" style="flex-direction:column;gap:4px">
        <div style="display:flex;gap:3px;flex-wrap:wrap">${dtags}</div>
        <div style="display:flex;gap:3px;flex-wrap:wrap">${wtags}</div>
      </div>
      <div class="cell" style="font-size:11px;color:#fff;line-height:1.6">${wstate}</div>
      <div class="cell r" style="font-family:monospace;font-size:11px;color:#fff">${r.scanned_at}</div>
    </div>`;
  }).join('');
  document.getElementById('tbl-body').innerHTML = rows;
}

let currentFilter = 'all';
function viewFilter(type) {
  currentFilter = type;
  document.querySelectorAll('.vbtn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('vb-' + type);
  if (btn) btn.classList.add('active');

  let filtered = [...allData];
  switch(type) {
    case 'regular':
      filtered = allData.filter(r => (r.regular_count||0) > 0); break;
    case 'hidden':
      filtered = allData.filter(r => (r.hidden_count||0) > 0); break;
    case 's7':
      filtered = allData.filter(r => (r.div_score||0) >= 7); break;
    case 's5':
      filtered = allData.filter(r => (r.div_score||0) >= 5); break;
    case 'up3':
      filtered = allData.filter(r => (r.change_pct||0) >= 3); break;
    case 'up':
      filtered = allData.filter(r => (r.change_pct||0) > 0); break;
    case 'best':
      filtered = allData.filter(r =>
        r.is_best || (r.today_red && r.above_ma5 && r.ma5_rising)
      ); break;
    case 'wmacd':
      filtered = allData.filter(r => r.weekly_macd_bull || r.macd_bull); break;
    case 'wkd':
      filtered = allData.filter(r => r.weekly_kd_cross); break;
    case 'macd':
      filtered = allData.filter(r => r.macd_bull); break;
    default:
      filtered = [...allData];
  }

  const countEl = document.getElementById('view-count');
  if (countEl) {
    const total = allData.length;
    countEl.textContent = type === 'all'
      ? `共 ${total} 支` : `${filtered.length} / ${total} 支`;
  }
  const isDW = scanMode === 'triple' || scanMode === 'resonance';
  if (isDW) renderDWTable(filtered);
  else      renderTable(filtered);
}

function showViewBar() {
  const bar = document.getElementById('view-bar');
  if (bar) {
    bar.style.display = 'flex';
    viewFilter('all');
  }
}

// ── 條件 A/B/C：7天累積表格 + 刪除 ──────────────────────
const COND_NAMES = {
  A: '條件A（三指標／日週雙重／日週共振）',
  B: '條件B（週威廉波浪 + 日週MACD共振）',
  C: '條件C（週威廉波浪 + 日威廉波浪）'
};

function startCondScan(cond) {
  setBtns(true);
  scanMode = 'cond' + cond;
  const label = COND_NAMES[cond] || ('條件' + cond);
  document.getElementById('stats-bar').style.display = 'none';
  document.getElementById('tbl-head').style.display  = 'none';
  document.getElementById('view-bar').style.display  = 'none';
  document.getElementById('scan-info').textContent = '';
  document.getElementById('prog').style.width = '0%';
  document.getElementById('mode-bar').innerHTML =
    `<span class="mode-tag" style="background:rgba(255,138,0,.18);color:#FFB74D;border:1px solid #FB8C00">● ${label} 掃描中...</span>`;
  document.getElementById('tbl-body').innerHTML =
    `<div class="empty"><span class="empty-icon" style="animation:blink 1s infinite">🗂️</span>
     <div class="empty-title" style="color:#FFB74D">${label} 掃描中...（第一次較慢，要補一年資料）</div>
     <div class="empty-sub" id="prog-txt">0 / 0 (0%)</div></div>`;

  clearInterval(progTimer);
  progTimer = setInterval(() => {
    fetch('/api/progress').then(r=>r.json()).then(d=>{
      if (d.total > 0) {
        const pct = Math.round(d.done / d.total * 100);
        document.getElementById('prog').style.width = pct + '%';
        const pt = document.getElementById('prog-txt');
        if (pt) pt.textContent = `${d.done} / ${d.total} (${pct}%)`;
        document.getElementById('scan-info').textContent = `${d.done}/${d.total}`;
        if (!d.running) clearInterval(progTimer);
      }
    }).catch(()=>{});
  }, 700);

  fetch(`/api/cond/${cond}/scan`)
    .then(r => r.json())
    .then(data => {
      clearInterval(progTimer);
      document.getElementById('prog').style.width = '100%';
      setTimeout(() => document.getElementById('prog').style.width = '0%', 1500);
      if (data.error) throw new Error(data.error);
      allData = data.data || [];
      document.getElementById('mode-bar').innerHTML =
        `<span class="mode-tag" style="background:rgba(255,138,0,.18);color:#FFB74D;border:1px solid #FB8C00">${label}</span>
         <span style="color:#3d5060;font-size:12px">
           共掃描 ${data.total_scanned} 支 · 今日命中 ${data.today_hits} 支 ·
           表內累積 ${data.table_count} 支（保留 ${data.keep_days} 天）· 耗時 ${data.elapsed} 秒
         </span>`;
      document.getElementById('scan-info').textContent =
        `✓ 今日 ${data.today_hits} · 表內 ${data.table_count} · ${data.elapsed}秒`;
      renderCondTable(allData, cond);
      setBtns(false);
    })
    .catch(e => {
      clearInterval(progTimer);
      document.getElementById('tbl-body').innerHTML =
        `<div class="empty"><span class="empty-icon">⚠</span>
         <div style="color:#ef5350">掃描失敗：${e.message}</div></div>`;
      document.getElementById('mode-bar').innerHTML =
        '<span class="mode-none" style="color:#ef5350">掃描失敗</span>';
      setBtns(false);
    });
}

function renderCondTable(rows, cond) {
  document.getElementById('tbl-head').style.display = 'none';
  if (!rows || rows.length === 0) {
    document.getElementById('tbl-body').innerHTML =
      `<div class="empty"><span class="empty-icon">🗂️</span>
       <div class="empty-title">條件${cond} 的 7 天表格目前是空的</div>
       <div class="empty-sub">今日沒有命中，過去 7 天也沒有累積（或都被刪除了）</div></div>`;
    return;
  }
  const body = rows.map((r, i) => {
    const up = (r.change_pct||0) > 0, zero = (r.change_pct||0) === 0;
    const cls = zero ? 'chg-zero' : (up ? 'chg-up' : 'chg-dn');
    const chgStr = (up?'▲':zero?'─':'▼') + ' ' + Math.abs(r.change||0).toFixed(2);
    const pctStr = (up?'+':'') + (r.change_pct||0).toFixed(2) + '%';
    const tags = (r.tags||[]).map(t => `<span class="cond-tag">${t}</span>`).join(' ');
    return `<tr>
      <td style="color:#4a6080;font-family:monospace">${i+1}</td>
      <td><span class="stock-id" style="font-size:17px">${r.id}　${r.name}</span></td>
      <td><span class="price" style="font-size:18px">${(r.close||0).toFixed(2)}</span></td>
      <td><span class="${cls}" style="font-size:16px">${chgStr}</span></td>
      <td><span class="${cls}" style="font-size:16px">${pctStr}</span></td>
      <td style="font-family:monospace;color:#fff">${r.vol_ratio||0}x</td>
      <td>${tags || '─'}</td>
      <td style="font-family:monospace;color:#7c8fa8;font-size:12px">${r.first_seen||''}</td>
      <td style="font-family:monospace;color:#9bb3cc;font-size:12px">${r.last_seen||''}</td>
      <td><button class="del-btn" onclick="deleteCond('${cond}','${r.id}')">🗑 刪除</button></td>
    </tr>`;
  }).join('');
  document.getElementById('tbl-body').innerHTML =
    `<table class="cond-tbl">
       <thead><tr>
         <th>#</th><th>股票</th><th>現價</th><th>漲跌</th><th>漲幅%</th><th>量比</th>
         <th>符合條件</th><th>首次入表</th><th>最後出現</th><th>操作</th>
       </tr></thead>
       <tbody>${body}</tbody>
     </table>`;
}

function deleteCond(cond, id) {
  if (!confirm(`確定要從條件${cond}的表格刪除 ${id} 嗎？`)) return;
  fetch(`/api/cond/${cond}/delete`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: id})
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) throw new Error(data.error);
      allData = data.data || [];
      document.getElementById('scan-info').textContent = `表內 ${data.table_count} 支`;
      renderCondTable(allData, cond);
    })
    .catch(e => alert('刪除失敗：' + e.message));
}

function exportCSV() {
  if (!allData.length) { alert('請先執行掃描'); return; }
  const header = ['代號','名稱','市場','模式','背離分','股價','漲跌','漲幅%',
                  '量比','RSI','KD','MACD方向','背離摘要','更新時間'];
  const rows = allData.map(r => [
    r.id, r.name, r.ex==='tse'?'上市':'上櫃',
    r.mode==='daily'?'日K':'週K',
    r.div_score, r.close, r.change, (r.change_pct||0).toFixed(2),
    r.vol_ratio, r.rsi, r.k_val,
    r.macd_bull?'多頭':'空頭',
    `"${r.div_summary}"`, r.scanned_at
  ].join(','));
  const csv  = '﻿' + header.join(',') + '\n' + rows.join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `背離選股_${scanMode==='daily'?'日K':scanMode==='weekly'?'週K':'KD'}_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── 啟動序列 ──────────────────────────────────────────────
def startup():
    print("  🚀 背離選股系統（極速版）啟動...")
    fast_warmup()          # ★ numba 預編譯
    load_delisted()
    _load_cond_tables()    # ★ 讀回 A/B/C 的 7 天累積表
    load_stock_list()
    batch_preload()
    incremental_refresh()  # ★ 自動偵測快取，必要時補最近 10 天


if __name__ == "__main__":
    threading.Thread(target=startup, daemon=True).start()
    print("""
╔══════════════════════════════════════════════════════╗
║   指標背離選股系統 v2（極速版） 日K / 週K 雙模式     ║
║   瀏覽器開啟：http://localhost:5001                  ║
║                                                      ║
║   優化：                                             ║
║   ✅ numba JIT pivot 偵測（20-50x）                  ║
║   ✅ 預擷取 numpy（去除 .iloc 開銷）                 ║
║   ✅ 批次平行預載（TSE+OTC 混合）                    ║
║   ✅ 週K resample 快取（日週雙掃免重算）             ║
╚══════════════════════════════════════════════════════╝""")
    app.run(host="0.0.0.0", port=PORT, debug=False)
