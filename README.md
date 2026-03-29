# Stock Sentiment Analyzer

Predicting Reliance Industries stock price movement using NLP sentiment analysis and Machine Learning.

---

## What This Project Does

This project combines **Natural Language Processing (NLP)** and **Machine Learning** to predict whether Reliance Industries (RELIANCE.NS) stock will move **UP or DOWN** the next trading day.

It works in 4 steps:
1. Downloads 1 year of real stock price data from Yahoo Finance
2. Scores financial news headlines using VADER sentiment analysis
3. Trains a Random Forest classifier on price + sentiment features
4. Outputs a prediction with confidence score and a price chart

---

## Demo Output

```
Model Accuracy: 45.65%

Tomorrow's Prediction for RELIANCE.NS: DOWN
Confidence: 86.00%
Average News Sentiment: 0.066

Chart saved as reliance_prediction.png
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core programming language |
| yfinance | Fetching historical stock data from Yahoo Finance |
| VADER (vaderSentiment) | NLP sentiment scoring of news headlines |
| pandas | Data manipulation and feature engineering |
| scikit-learn | Random Forest ML model + evaluation |
| matplotlib | Price chart visualization |
| VS Code | Development environment |

---

## How to Set Up and Run

### Prerequisites
- Python 3.10 or higher
- VS Code with Python extension

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-sentiment-analyser.git
cd stock-sentiment-analyser
```

### Step 2 — Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install yfinance vaderSentiment pandas scikit-learn matplotlib
```

### Step 4 — Run the project

```bash
python main.py
```

A chart window will open and `reliance_prediction.png` will be saved in the project folder.

---

## How It Works

### 1. Data Collection
Uses `yfinance` to download 1 year of daily closing prices for RELIANCE.NS. A binary target label is created: `1` if tomorrow's price is higher than today's (UP), `0` if lower (DOWN).

### 2. Sentiment Analysis
Financial news headlines are scored using VADER NLP. VADER returns a compound score from `-1.0` (most negative) to `+1.0` (most positive). The average score across all headlines is used as a feature.

### 3. Feature Engineering
Four features are passed to the ML model:
- `Price_Change` — daily % change in closing price
- `MA_5` — 5-day moving average
- `MA_20` — 20-day moving average
- `Sentiment` — average VADER compound score

### 4. Model Training
A Random Forest Classifier (100 trees) is trained on an 80/20 train/test split. The model learns patterns between the features and next-day price direction.

### 5. Prediction
The model predicts UP or DOWN for the next trading day using the most recent data point, along with a confidence probability.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 45.65% |
| UP Precision | 0.50 |
| UP Recall | 0.52 |
| DOWN Precision | 0.40 |
| DOWN Recall | 0.38 |
| Test Set Size | 46 trading days |

> Note: Stock prediction is inherently difficult. Even professional quantitative models struggle to exceed 55-60% accuracy on short-term daily predictions. The goal of this project is to demonstrate the complete ML pipeline, not to guarantee financial returns.

---

## Known Limitations

- Sentiment is computed from static sample headlines, not live scraped news
- Only one stock (Reliance Industries) is analyzed
- No macroeconomic indicators (interest rates, sector indices) are included
- The model is retrained from scratch each run (no model persistence)

---

## Future Improvements

- Live news scraping from Yahoo Finance RSS or NewsAPI
- FinBERT (financial NLP transformer) instead of VADER
- Additional features: RSI, MACD, trading volume
- Streamlit web dashboard for interactive use
- Backtesting module to simulate trading performance
- Support for multiple stock tickers

---

## Course Context

This project was built as a **Bring Your Own Project (BYOP)** capstone for the AI/ML course. It applies course concepts including supervised learning, classification, NLP, feature engineering, and model evaluation to a self-identified real-world problem.

---

## License

MIT License — free to use, modify, and distribute.
