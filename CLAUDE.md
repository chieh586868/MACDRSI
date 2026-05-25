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

### macd_app.py（盤中選股，條件 A~G）
A/B/C 共用「穩健門檻 common」：量比≥1.2、漲幅≥1%、現價站上短均、UT在追蹤線上、MA鬆散。
- A：UT買訊 + 日威廉波浪 + 指標背離 + common
- B：UT + 費波南(站上短均) + 週威廉波浪 + common
- C：UT + 費波南 + 日威廉 + 確認分>4 + common
- D（背離）：嚴格日週雙重背離（日K背離≥5 + 週K背離≥2）
- E（MACD共振）：日MACD金叉 + 站上MA20 + 量比≥1.5 + 週MACD多頭 + 評分≥6 + 離MA20乖離≤12% + 昨日回檔(昨收≤昨MA5或MA10)
- F（週威廉波浪）：週威廉波浪完成 + 符合 A/B/C/D/E 任一 + 昨日回檔
- G（綜合，簡化版）：群1(UT或費波南) + 昨日回檔 + 今日突破變盤訊號≥4 ＝ 買點⚡
- 掃描範圍：台股上市+上櫃全部（約1900支），可勾「排除冷門產業」（水泥/塑膠/紡織/營建/航運/造紙/觀光/橡膠）

### divergence_screener.py（指標背離）
- 日K背離 / 週K背離 / KD背離：MACD/RSI/KD 底背離，分數制
- 三指標+日週雙重 / 日週強烈共振：沿用本程式既有 double/strong resonance
- A：三指標背離 **或** 日週雙重背離(日≥5且週≥2) **或** 日週共振(日≥5且週≥4)（OR）→ 7天累積表
- B：週威廉波浪 + 日週MACD共振（日DIF>DEA 且 週DIF>DEA）→ 7天累積表
- C：週威廉波浪 + 日威廉波浪 → 7天累積表
- 掃描範圍：tw_stocks.json 全清單

## 介面元件

### macd_app.py（localhost:5000）
- 條件按鈕：條件A / 條件B / 條件C / 背離D / MACD共振E / 週威廉波浪F / 綜合G
- 自動掃描（每N分鐘）啟動/停止、Telegram 通知開關+測試
- 「排除冷門產業」勾選、掃描結果再過濾 A~G、匯出 CSV
- 結果區：股號/名稱/現價/漲跌/漲幅/量比…；G 有「買點⚡ 今X」徽章（簡化後不再有「/周X」）

### divergence_screener.py（localhost:5001）
- 條件按鈕：🟠條件A / 🔵條件B / 🟢條件C（各為 7 天累積表，每列有 🗑 刪除按鈕）
- 原有按鈕：🔴三指標+日週雙重 / 🔮日週強烈共振 / 📥匯出CSV
- 進度條、統計列、快速檢視過濾列
- A/B/C 結果區（獨立表格）：# / 股票 / 現價 / 漲跌 / 漲幅% / 量比 / 符合條件 / 首次入表 / 最後出現 / 刪除

## 目前已知問題
- [x] 「沒有抓到任何股票」→ 已釐清：多半是條件太嚴（如 G 為最嚴的綜合條件），非資料/後端問題；另「Failed to fetch」是把 index.html 用檔案(file://)開啟、未透過 http://localhost:5000，改用網址列開即正常。
- [x] G 按鈕刷新後行為 → 已確認：重啟後端 + Ctrl+Shift+R 強制刷新後，G 鈕與過濾列正常回復。

## 待辦 / 下一步
- [ ] **(資安・優先)** 至永豐後台作廢並重新產生 Shioaji API 金鑰（舊金鑰曾外洩於 git 歷史 commit c2486db）
- [ ] macd_app 的 `index.html` 尚未進 repo，前端若要改動需先 push 上來
- [ ] divergence A/B/C：若要「刪除後永久不再出現」，需加「已忽略清單」（目前刪除後再掃到仍會回來）
- [ ] 確認 B/C 的週威廉波浪參數是否要與 macd_app（F 的 preset）完全一致
- [ ] divergence_screener 的 A/B/C 第一次掃描較慢（補一年資料），之後走 7 天快取

## 重要決策紀錄
- 2026/5/25 divergence_screener 新增條件 A/B/C，各一張「7天累積表」（超過7天自動移除、可手動刪除，存 cond_tables.json）。A=三指標背離 / 日週雙重 / 日週共振（**OR** 任一即入表）；B=週威廉波浪+日週MACD共振（日DIF>DEA 且 週DIF>DEA）；C=週威廉波浪+日威廉波浪。威廉用 williams_v4。
- 2026/5/25 macd_app 條件G 簡化為「群1(UT/費波南)+昨日回檔+今日突破≥4=買點」，移除群2/昨日變盤<4/週變盤三層。
- 2026/5/25 macd_app 條件E 加「昨日回檔」過濾（1日前收盤 ≤ 1日前MA5 或 ≤ 1日前MA10）。
- 2026/5/25 移除 wr_scan.py（含外洩的永豐 API 金鑰）；金鑰已在 git 歷史，**需至永豐後台作廢並重新產生**。
