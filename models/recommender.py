import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
dest = pd.read_csv("data/raw/processed/clean_destinations.csv")

# Fill missing values
dest.fillna("", inplace=True)

# Combine correct columns
dest["features"] = (
    dest["Type"] + " " +
    dest["Significance"] + " " +
    dest["State"]
)

# Convert text to numbers
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(dest["features"])

# Compute similarity
similarity = cosine_similarity(feature_matrix)

# Recommendation function
def recommend(place, top_n=5):
    try:
        index = dest[dest["Name"] == place].index[0]
        state = dest.iloc[index]["State"]

        scores = list(enumerate(similarity[index]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        results = []

        for i in sorted_scores:
            place_name = dest.iloc[i[0]]["Name"]
            place_state = dest.iloc[i[0]]["State"]

            if place_name != place and place_state == state:
                results.append(place_name)

        return results[:top_n]

    except:
        return ["No recommendations found"]