import shioaji as sj
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import time

API_KEY    = "Bpf6McAWj2VRtTmKVRV5vSAjyj5SXJwjqaBeCHgD7qEG"
SECRET_KEY = "Aqd5ZKcv7eH6cpWcSjEBRbc3Nk9FPVMYUJdQkYbzGHpz"

SYMBOLS = [
    "2330","2317","2382","2308","2454","3711","2379","2412",
    "2881","2882","2603","2615","2385","6505","1301","3034",
    "2303","2357","2376","2395","3008","2301","2408","2409",
    "1216","2912","2207","2002","1303","1326","2886","2884",
    "2891","2883","5880","2890","2887","2885","2880","2888"
]

def williams_r(df, period=20):
    hh = df["High"].rolling(period).max()
    ll = df["Low"].rolling(period).min()
    return (hh - df["Close"]) / (hh - ll) * -100

def detect_wave5(df, symbol):
    df = df.copy()
    df["wr"] = williams_r(df)
    df = df.dropna(subset=["wr"])
    if len(df) < 30:
        return None
    w = df.tail(80)["wr"].values
    peaks   = argrelextrema(w, np.greater, order=1)[0]
    troughs = argrelextrema(w, np.less,    order=1)[0]
    pivots  = sorted([(i,"P",w[i]) for i in peaks] +
                     [(i,"T",w[i]) for i in troughs])
    if len(pivots) < 4:
        return None
    last4 = pivots[-4:]
    types = [x[1] for x in last4]
    now   = w[-1]
    if types == ["P","T","P","T"] and now < last4[-1][2]:
        return {"symbol":symbol, "dir":"BEAR W5", "wr":round(now,1)}
    if types == ["T","P","T","P"] and now > last4[-1][2]:
        return {"symbol":symbol, "dir":"BULL W5", "wr":round(now,1)}
    return None

print("Connecting...")
api = sj.Shioaji(simulation=False)
api.login(api_key=API_KEY, secret_key=SECRET_KEY)
time.sleep(5)

results = []
for sym in SYMBOLS:
    try:
        kb = api.kbars(
            contract=api.Contracts.Stocks[sym],
            start="2025-10-01",
            end=""%Y-%m-%d"",
            timeout=30000
        )
        df = pd.DataFrame({**kb})
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df[df["ts"] < 2000000000]
        df["date"] = pd.to_datetime(df["ts"], unit="s").dt.normalize()
        df = df.groupby("date").agg(
            Open=("Open","first"),
            High=("High","max"),
            Low=("Low","min"),
            Close=("Close","last")
        ).reset_index()
        r = detect_wave5(df, sym)
        if r:
            results.append(r)
            print(f"[HIT] {r['symbol']}  {r['dir']}  WR={r['wr']}")
        else:
            print(f"[ - ] {sym}")
        time.sleep(0.2)
    except Exception as e:
        print(f"[ERR] {sym}: {e}")

api.logout()
print("\n=== RESULT ===")
if results:
    for r in sorted(results, key=lambda x: x["wr"]):
        print(f"{r['symbol']}  {r['dir']}  WR={r['wr']}")
else:
    print("No stocks found")