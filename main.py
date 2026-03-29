import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Part 1: Fetch stock data 
ticker = "RELIANCE.NS"
stock = yf.download(ticker, period="1y", interval="1d", progress=False)
stock.columns = stock.columns.get_level_values(0)
stock = stock[["Close"]].copy()
stock["Target"] = (stock["Close"].shift(-1) > stock["Close"]).astype(int)

# Part 2: Sentiment scoring 
analyzer = SentimentIntensityAnalyzer()

headlines = [
    "Reliance Industries reports record quarterly profit",
    "Reliance Jio adds 10 million new subscribers this month",
    "Reliance stock falls amid global market selloff",
    "Reliance expands green energy investment to 10 billion dollars",
    "Reliance faces regulatory scrutiny over retail expansions",
    "Mukesh Ambani announces major partnership with global tech firm",
    "Reliance Industries beats analyst expectations for third quarters",
    "Oil prices drop impacting Reliance refinery margins",
]

scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
avg_sentiment = sum(scores) / len(scores)

# Part 3: Training model 
stock["Price_Change"] = stock["Close"].pct_change()
stock["MA_5"] = stock["Close"].rolling(window=5).mean()
stock["MA_20"] = stock["Close"].rolling(window=20).mean()
stock["Sentiment"] = avg_sentiment
stock = stock.dropna()
stock = stock[:-1]

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
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Part 4: Live prediction and chart 
# Predict tomorrow using the most recent data point
latest = stock[features].iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest)[0]
confidence = model.predict_proba(latest)[0][prediction] * 100

direction = "UP" if prediction == 1 else "DOWN"
print(f"\nTomorrow's Prediction for RELIANCE.NS: {direction}")
print(f"Confidence: {confidence:.2f}%")
print(f"Average News Sentiment: {avg_sentiment:.3f}")

# Plot closing price with moving averages
plt.figure(figsize=(12, 6))
plt.plot(stock.index, stock["Close"], label="Close Price", color="#1D9E75", linewidth=1.5)
plt.plot(stock.index, stock["MA_5"], label="5-Day MA", color="#EF9F27", linewidth=1, linestyle="--")
plt.plot(stock.index, stock["MA_20"], label="20-Day MA", color="#7F77DD", linewidth=1, linestyle="--")

plt.title(f"RELIANCE.NS — 1 Year Price + Prediction: {direction} ({confidence:.1f}% confidence)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.tight_layout()
plt.savefig("reliance_prediction.png", dpi=150)
plt.show()
print("\nChart saved as reliance_prediction.png")

#Project Idea and Executability by Sanket Suri (25BAI10680)
