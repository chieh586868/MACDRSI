# MACDRSI 技術分析選股工具

支援台股（TWSE/OTC）與美股（NYSE/NASDAQ），使用 MACD、RSI、KD、均線等技術指標篩選強勢股。

## 安裝

```bash
pip install -r requirements.txt
```

## 快速使用

```bash
# 掃描全部（台股 + 美股），預設策略
python main.py

# 只掃台股
python main.py --market tw

# 只掃美股，使用突破策略
python main.py --market us --strategy breakout

# 指定特定股票
python main.py --symbols 2330.TW AAPL NVDA

# 輸出 CSV
python main.py --market both --output result.csv
```

## 策略說明

| 策略 | 說明 | RSI 範圍 | 適合 |
|------|------|----------|------|
| `default` | 均衡策略：MACD 多頭 + 均線多頭 + 價>MA20 | 30~70 | 一般趨勢跟蹤 |
| `breakout` | 突破策略：MACD 金叉 + 量增 + 均線多頭 | 50~80 | 強勢突破股 |
| `oversold` | 超賣反彈：RSI 低檔 + KD 金叉 | 20~40 | 短線反彈 |

## 技術指標

- **MACD**（12/26/9）：快慢線關係、Histogram 方向
- **RSI**（14）：超買超賣判斷
- **KD**（9）：隨機指標金叉訊號
- **均線**：MA5、MA20、MA60 多頭排列

## 自訂股票清單

編輯 `stocks/tw_stocks.txt` 或 `stocks/us_stocks.txt`，每行一個代碼：

```
# 台股加 .TW 後綴
2330.TW
2454.TW

# 美股直接填代碼
NVDA
MSFT
```

## 自訂 RSI 範圍

```bash
python main.py --rsi-min 40 --rsi-max 65
```
