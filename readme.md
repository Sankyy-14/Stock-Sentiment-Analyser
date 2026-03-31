# Stock Sentiment Analyzer

A Python tool that predicts whether an Indian stock will go UP or DOWN the next trading day — using live news sentiment and machine learning.

---

## What It Does

You type in any NSE/BSE stock ticker. The program:
1. Downloads 1 year of real price data from Yahoo Finance
2. Fetches today's live news headlines from Google News
3. Scores the headlines using VADER sentiment analysis
4. Trains an XGBoost classifier on price + sentiment features
5. Predicts tomorrow's direction with a confidence score
6. Saves a price chart to your folder

---

## Demo

```
Enter stock ticker (e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS): RELIANCE.NS

Stock data ready for RELIANCE.NS!

Fetched 10 live headlines:
  +0.296 | Reliance Industries shares slump 4%, m-cap slips by ₹80,000 crore - Upstox
  -0.660 | Stock in focus: Reliance share price to be on radar after Iranian crude oil denial - Mint
  +0.361 | Reliance shares drop 5%, wipe off Rs 88,000 cr from market value - Economic Times
  ...

Average sentiment score: 0.117
Training on 227 trading days of data
Model Accuracy: 54.35%

Tomorrow's Prediction for RELIANCE.NS: DOWN
Confidence: 76.00%
Average News Sentiment: 0.117

Chart saved as RELIANCE_NS_prediction.png
```

---

## Supported Tickers

Any NSE or BSE listed stock. Some examples:

| Company | Ticker |
|---|---|
| Reliance Industries | RELIANCE.NS |
| TCS | TCS.NS |
| HDFC Bank | HDFCBANK.NS |
| Infosys | INFY.NS |
| Wipro | WIPRO.NS |
| State Bank of India | SBIN.NS |

---

## Tech Stack

- **Python 3.13**
- **yfinance** — live stock price data from Yahoo Finance
- **feedparser** — live news headlines from Google News RSS
- **vaderSentiment** — NLP sentiment scoring
- **pandas** — data manipulation and feature engineering
- **XGBoost** — ML classifier for UP/DOWN prediction
- **scikit-learn** — train/test split and evaluation metrics
- **matplotlib** — price chart with moving averages

---

## How to Set Up

### 1. Clone the repo

```bash
git clone https://github.com/Sankyy-14/Stock-Sentiment-Analyser.git
cd Stock-Sentiment-Analyser
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install yfinance feedparser vaderSentiment pandas xgboost scikit-learn matplotlib
```

### 4. Run

```bash
python main.py
```

Enter any NSE ticker when prompted and the prediction will run.

---

## How It Works

### Live news scraping
Instead of static headlines, the program fetches today's actual news for the stock you enter using Google News RSS. No API key needed.

### Sentiment scoring
Each headline gets a compound score from -1.0 (very negative) to +1.0 (very positive) using VADER NLP. The average across all headlines becomes one of the model's features.

### Feature engineering
Four features are passed to the model:
- `Price_Change` — daily % change in closing price
- `MA_5` — 5-day moving average
- `MA_20` — 20-day moving average
- `Sentiment` — average VADER score from today's headlines

### XGBoost classifier
Trained on an 80/20 split of 1 year of daily data. XGBoost is the industry standard for tabular financial data and outperforms Random Forest on this task.

### Output
- Terminal: prediction (UP/DOWN), confidence %, sentiment score, full classification report
- File: `{TICKER}_prediction.png` — price chart with 5-day and 20-day moving averages

---

## Model Performance

Tested on RELIANCE.NS with live news:

| Version | Sentiment Source | Accuracy |
|---|---|---|
| v1 | Static headlines | 45.65% |
| v2 | Live Google News | 54.35% |
| v3 | Live + XGBoost | TBD |

> Stock prediction is hard. Even professional quant systems rarely exceed 55-60% on short-term daily predictions. The goal here is a working end-to-end pipeline, not a trading bot.

---

## Project Structure

```
Stock-Sentiment-Analyser/
├── main.py                         # Main program
├── RELIANCE_NS_prediction.png      # Sample chart output
├── BYOP_Project_Report_Sanket.docx # Project report
└── README.md
```

---

## Known Limitations

- Sentiment is based on headlines only, not full article content
- One model trained fresh per run — no model persistence yet
- Accuracy varies by stock and market conditions

---

## What's Next

- [ ] Backtesting simulator — did predictions make money over the past year?
- [ ] Streamlit web dashboard — run it in a browser, no terminal needed
- [ ] FinBERT — financial NLP transformer for better sentiment accuracy

---

## Course Context

Built as a BYOP capstone for an AI/ML course. Started as a basic Random Forest classifier on static data and grew into a live news-driven prediction tool.

---

## License

MIT — free to use and build on.
