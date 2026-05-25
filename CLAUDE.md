# CLAUDE.md — MACDRSI 選股系統

## 專案目的
用 MACD 與 RSI 指標篩選股票的選股系統。
前端有操作介面,後端負責抓資料、計算指標、回傳符合條件的股票清單。

## 技術架構
- 前端:純 HTML+JS（內嵌在各自的 .py，透過 Flask `render_template_string` 提供）
- 後端:Python Flask
- 資料來源:`divergence_screener.py` 用 yfinance；`macd_app.py` 用永豐 Shioaji 即時（無則 TWSE 延遲20分）
- 執行方式:
  - `python macd_app.py` → http://localhost:5000（MACD/RSI 盤中選股，條件 A~G）
  - `python divergence_screener.py` → http://localhost:5001（指標背離選股，條件 A/B/C + 三指標/日週雙重/共振）

## 選股邏輯（目前設定）
- MACD 條件:（例如 MACD 黃金交叉 / DIF 上穿 DEA）
- RSI 條件:（例如 RSI < 30 或 RSI 由下往上突破 50）
- 兩者關係:（AND 同時成立 / OR 擇一）
- 掃描範圍:（例如台股上市全部 / 某清單）

## 介面元件
- 「G」按鈕:用途是（填,例如觸發掃描 / 刷新結果）
- 結果區:顯示掃出的股票清單

## 目前已知問題
- [ ] 掃描後「沒有抓到任何股票」,需確認是條件太嚴、資料沒抓到、還是後端沒回傳
- [ ] G 按鈕刷新後行為待確認

## 待辦 / 下一步
- [ ] 

## 重要決策紀錄
- 2026/5/25 divergence_screener 新增條件 A/B/C，各一張「7天累積表」（超過7天自動移除、可手動刪除，存 cond_tables.json）。A=三指標背離 / 日週雙重 / 日週共振（**OR** 任一即入表）；B=週威廉波浪+日週MACD共振（日DIF>DEA 且 週DIF>DEA）；C=週威廉波浪+日威廉波浪。威廉用 williams_v4。
- 2026/5/25 macd_app 條件G 簡化為「群1(UT/費波南)+昨日回檔+今日突破≥4=買點」，移除群2/昨日變盤<4/週變盤三層。
- 2026/5/25 macd_app 條件E 加「昨日回檔」過濾（1日前收盤 ≤ 1日前MA5 或 ≤ 1日前MA10）。
- 2026/5/25 移除 wr_scan.py（含外洩的永豐 API 金鑰）；金鑰已在 git 歷史，**需至永豐後台作廢並重新產生**。
