import streamlit as st
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

st.title("🌍 Tourism Data Analytics Dashboard")

menu = st.sidebar.selectbox(
    "Select Section",
    ["Destinations Analysis",
     "Hotels Analysis",
     "Sentiment Analysis",
     "Price Prediction",
     "Monument Images"]
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
    st.header("Hotel Price Prediction")
    
    rating = st.slider(
        "Hotel Rating",
        1.0, 5.0, 4.0
    )
    price = price_model.predict([[rating]])[0]

    st.success(f"Estimated Price ₹ {round(float(price),2)}")

    
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


        
