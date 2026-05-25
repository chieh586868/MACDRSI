# MACDRSI 選股系統

> **主程式：`macd_app.py`**（MACD / RSI 盤中選股系統）。
> 之後所有 MACD 盤中系統的修改都只動 `macd_app.py`，**請勿修改 `app_optimized.py`**。

## 執行方式

- **主程式**：`python macd_app.py` → http://localhost:5000
  - 盤中選股，條件 A~G（UT 買訊 / 費波南 / 威廉波浪 / 指標背離 / MACD 共振 / 週威廉波浪 / 綜合買點⚡）
- 輔助程式：`python divergence_screener.py` → http://localhost:5001
  - 指標背離選股（MACD / RSI / KD 底背離 + 條件 A/B/C 的 7 天累積表，可排除冷門產業）

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `macd_app.py` | ★ **主程式**：MACD/RSI 盤中選股（localhost:5000） |
| `divergence_screener.py` | 指標背離選股系統（localhost:5001） |
| `fast_indicators.py` | numba 加速的指標計算 |
| `williams_v4.py` | 威廉波浪偵測 |
| `CLAUDE.md` | 專案說明、選股邏輯、開發約定、決策紀錄 |

## 資料來源

- `macd_app.py`：永豐 Shioaji 即時報價（無則 TWSE 延遲 20 分）
- `divergence_screener.py`：yfinance

詳細的選股邏輯、介面元件、開發約定與決策紀錄，請見 [`CLAUDE.md`](./CLAUDE.md)。
