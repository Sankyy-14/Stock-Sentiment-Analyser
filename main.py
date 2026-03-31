import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Ticker input 
ticker = input("Enter stock ticker (e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS): ").strip().upper()
if not ticker:
    ticker = "RELIANCE.NS"
    print("No ticker entered, defaulting to RELIANCE.NS")

# Part 1: Fetch stock data 
stock = yf.download(ticker, period="1y", interval="1d", progress=False)
stock.columns = stock.columns.get_level_values(0)
stock = stock[["Close"]].copy()
stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)

print(f"\nStock data ready for {ticker}!")

# Part 2: Live news sentiment 
analyzer = SentimentIntensityAnalyzer()

query = ticker.replace(".NS", "").replace(".BO", "")
rss_url = f"https://news.google.com/rss/search?q={query}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

headlines = [entry.title for entry in feed.entries[:10]]

if not headlines:
    print("No live headlines found, using fallback.")
    headlines = [f"{query} stock steady amid market activity"]

print(f"\nFetched {len(headlines)} live headlines:")
for h in headlines:
    score = analyzer.polarity_scores(h)["compound"]
    print(f"  {score:+.3f} | {h}")

scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
avg_sentiment = sum(scores) / len(scores)
print(f"\nAverage sentiment score: {avg_sentiment:.3f}")

# Part 3: Feature engineering + train model 
stock["Price_Change"] = stock["Close"].pct_change()
stock["MA_5"] = stock["Close"].rolling(window=5).mean()
stock["MA_20"] = stock["Close"].rolling(window=20).mean()
stock["Sentiment"] = avg_sentiment
stock = stock.dropna()
stock = stock[:-1]

print(f"Training on {len(stock)} trading days of data")

features = ["Price_Change", "MA_5", "MA_20", "Sentiment"]
X = stock[features]
y = stock["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=["DOWN", "UP"]))

# Part 4: Prediction + chart 
latest = stock[features].iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest)[0]
confidence = model.predict_proba(latest)[0][prediction] * 100

direction = "UP" if prediction == 1 else "DOWN"
print(f"Tomorrow's Prediction for {ticker}: {direction}")
print(f"Confidence: {confidence:.2f}%")
print(f"Average News Sentiment: {avg_sentiment:.3f}")

chart_filename = f"{ticker.replace('.', '_')}_prediction.png"
plt.figure(figsize=(12, 6))
plt.plot(stock.index, stock["Close"], label="Close Price", color="#1D9E75", linewidth=1.5)
plt.plot(stock.index, stock["MA_5"], label="5-Day MA", color="#EF9F27", linewidth=1, linestyle="--")
plt.plot(stock.index, stock["MA_20"], label="20-Day MA", color="#7F77DD", linewidth=1, linestyle="--")
plt.title(f"{ticker} — 1 Year Price + Prediction: {direction} ({confidence:.1f}% confidence)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.tight_layout()
plt.savefig(chart_filename, dpi=150)
plt.show()
print(f"\nChart saved as {chart_filename}")