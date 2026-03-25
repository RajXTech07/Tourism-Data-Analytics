import streamlit as st
from models.recommender import recommend
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(
    page_title="Toursim Analytics Dashboard",
    layout = "wide"
)

dest = pd.read_csv("data/raw/processed/clean_destinations.csv")
hotel = pd.read_csv("data/raw/processed/clean_hotels.csv")


price_model = pickle.load(open(
    "models/hotel_price_model.pkl","rb"
))

sentiment_model = pickle.load(open(
    "models/sentiment_model.pkl","rb"
))

vectorizer = pickle.load(open(
    "models/vectorizer.pkl","rb"
))
#Pkl file.....
le_city = pickle.load(open("models/le_city.pkl", "rb"))
le_type = pickle.load(open("models/le_type.pkl", "rb"))

#Features columns
feature_cols = [col for col in hotel.columns if "Feature" in col]

hotel["all_features"] = hotel[feature_cols].astype(str).agg(" ".join, axis=1)

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

st.title("🌍 Tourism Data Analytics Dashboard")

menu = st.sidebar.selectbox(
    "Select Section",
    ["Destinations Analysis",
     "Hotels Analysis",
     "Sentiment Analysis",
     "Price Prediction",
     "Monument Images",
     "Recommendation System"]
)

#DESTINATIONS ANALYSIS
if menu == "Destinations Analysis":
    st.header("Top rated destinations")
    
    top_dest = dest.sort_values(
        "Google review rating",
        ascending=False
    ).head(10)
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(top_dest["Name"],
            top_dest["Google review rating"])
    ax.invert_yaxis()
    st.pyplot(fig)
    
    state = st.selectbox(
        "Filter State",
        dest["State"].unique()
    )
    
    st.dataframe(dest[dest["State"]==state])
    
## Hotel Analysis
elif menu == "Hotels Analysis":
    st.header("Hotel Price Distribution")
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(hotel["Hotel_Price"], bins=20)
    st.pyplot(fig)
    
    st.subheader("Top Expensive Hotels")
    st.dataframe(
        hotel.sort_values("Hotel_Price",
                          ascending=False).head(10)
    )

## Sentiment Analysis
elif menu == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    review = st.text_area("Enter your review")
    if st.button("Predict"):
        vec = vectorizer.transform([review])
        result = sentiment_model.predict(vec)[0]
        
        if result == 1:
            st.success("Positive Review😍")
        else:
            st.error("Negative Review🤬")
            
            
## Price Predicition

elif menu == "Price Prediction":
    st.header("🏨 Hotel Price Prediction")

    # Inputs
    rating = st.slider("Hotel Rating", 1.0, 5.0, 4.0)

    city = st.selectbox("Select City", hotel["City"].unique())
    hotel_type = st.selectbox("Select Hotel Type", hotel["Hotel_Type"].unique())

    if st.button("Predict Price"):
        try:
            # Encode inputs
            city_encoded = le_city.transform([city])[0]
            type_encoded = le_type.transform([hotel_type])[0]

            # Prediction
            price = price_model.predict([[rating, city_encoded, type_encoded]])[0]

            st.success(f"Estimated Price ₹ {round(float(price), 2)}")

        except Exception as e:
            st.error(f"Error: {e}")

    
# Monument Images
elif menu == "Monument Images":

    import os

    st.header("🗺 Indian Monument Gallery")

    base_folder = "data/raw/Indian-monuments/images/test"

    if not os.path.exists(base_folder):
        st.error("Image folder not found!")

    else:

        # ✅ Get all monument folder names
        monument_names = [
            folder for folder in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, folder))
        ]

        monument_names.sort()

        # ✅ Selectbox
        selected_monument = st.selectbox(
            "Select Monument",
            monument_names
        )

        monument_path = os.path.join(base_folder, selected_monument)

        images = [
            img for img in os.listdir(monument_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        st.success(
            f"Showing {len(images)} images of {selected_monument}"
        )

        # ✅ Display Images
        cols = st.columns(4)

        for i, img in enumerate(images):

            img_path = os.path.join(monument_path, img)

            with cols[i % 4]:
                st.image(
                    img_path,
                    caption=selected_monument,
                    use_container_width=True
                )

#Recommendation Dashboard
elif menu == "Recommendation System":

    st.title("🎯 AI Destination Recommendation")

    place = st.selectbox("Select a Destination", dest["Name"].unique())

    if st.button("Get Recommendations"):
        results = recommend(place)

        st.subheader("Recommended Places:")
        for r in results:
            st.write("👉", r)


        
