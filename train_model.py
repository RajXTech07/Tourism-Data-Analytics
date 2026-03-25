import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("models", exist_ok=True)


print("Traning model...")

# Datasets....
hotel = pd.read_csv("data/raw/processed/clean_hotels.csv")
reviews = pd.read_csv("data/raw/processed/clean_reviews.csv")

# HOTEL PRICE MODEL
X = hotel[["Hotel_Rating"]]
y = hotel["Hotel_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

price_model = LinearRegression()
price_model.fit(X_train, y_train)

pickle.dump(price_model,
            open("models/hotel_price_model.pkl","wb"))

print("Hotel price model saved!")

# SENTIMENT MODEL
reviews["sentiment"] = (reviews["Rating"]>=3).astype(int)


vectorizer = TfidfVectorizer(stop_words="english")
X_text = vectorizer.fit_transform(reviews["Review"])
y_text = reviews["sentiment"]

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_text, y_text)

pickle.dump(sentiment_model,
            open("models/sentiment_model.pkl","wb"))

pickle.dump(vectorizer,
            open("models/vectorizer.pkl","wb"))

print("Sentiment model saved!")