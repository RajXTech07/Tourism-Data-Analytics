import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Create models folder
os.makedirs("models", exist_ok=True)

print("Training model...")


# LOAD DATA

hotel = pd.read_csv("data/raw/processed/clean_hotels.csv")
reviews = pd.read_csv("data/raw/processed/clean_reviews.csv")


# Combine all feature columns
feature_cols = [col for col in hotel.columns if "Feature" in col]

hotel["all_features"] = hotel[feature_cols].astype(str).agg(" ".join, axis=1)

# Extract hotel type
def extract_type(text):
    text = text.lower()
    if "5-star" in text:
        return "5-star"
    elif "4-star" in text:
        return "4-star"
    elif "3-star" in text:
        return "3-star"
    else:
        return "other"

hotel["Hotel_Type"] = hotel["all_features"].apply(extract_type)

# ENCODING
le_city = LabelEncoder()
le_type = LabelEncoder()

hotel["City"] = le_city.fit_transform(hotel["City"])
hotel["Hotel_Type"] = le_type.fit_transform(hotel["Hotel_Type"])

# MODEL TRAINING (PRICE)
X = hotel[["Hotel_Rating", "City", "Hotel_Type"]]
y = hotel["Hotel_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

price_model = LinearRegression()
price_model.fit(X_train, y_train)

# Save model
with open("models/hotel_price_model.pkl", "wb") as f:
    pickle.dump(price_model, f)

# Save encoders
with open("models/le_city.pkl", "wb") as f:
    pickle.dump(le_city, f)

with open("models/le_type.pkl", "wb") as f:
    pickle.dump(le_type, f)

print("Hotel price model saved!")

# SENTIMENT MODEL
reviews["sentiment"] = (reviews["Rating"] >= 3).astype(int)

vectorizer = TfidfVectorizer(stop_words="english")
X_text = vectorizer.fit_transform(reviews["Review"])
y_text = reviews["sentiment"]

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_text, y_text)

# Save sentiment model
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)

# Save vectorizer
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Sentiment model saved!")