# ===========================================
# COVID-19 Vaccine Sentiment Analysis
# Streamlit Web App
# ===========================================

import streamlit as st
import joblib
import re
import string
import nltk
import pandas as pd


from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# ===========================================
# Page Configuration
# ===========================================

st.set_page_config(
    page_title="COVID-19 Vaccine Sentiment Analysis",
    page_icon="💉",
    layout="wide"
)

# ===========================================
# Load CSS
# ===========================================

with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

# ===========================================
# Load ML Model
# ===========================================

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# ===========================================
# Stopwords
# ===========================================

stop_words = set(stopwords.words("english"))

# ===========================================
# Text Cleaning Function
# ===========================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol only
    text = text.replace("#", "")

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = " ".join(text.split())

    # Remove stopwords
    text = " ".join([
        word for word in text.split()
        if word not in stop_words
    ])

    return text
# ===========================================
# SIDEBAR
# ===========================================

st.sidebar.title("💉 Vaccine Sentiment Analysis")

st.sidebar.markdown("---")

st.sidebar.write("### 📋 Project")

st.sidebar.write("""
This application predicts the sentiment of
COVID-19 vaccine related tweets using
Machine Learning.
""")

st.sidebar.markdown("---")

st.sidebar.write("### 🤖 Model")

st.sidebar.success("Logistic Regression")

st.sidebar.markdown("---")

st.sidebar.write("Developed using")

st.sidebar.write("✔ Python")

st.sidebar.write("✔ Streamlit")

st.sidebar.write("✔ Scikit-Learn")

st.sidebar.write("✔ TF-IDF")


# ===========================================
# MAIN TITLE
# ===========================================

st.markdown("""
<h1 style='text-align:center;
color:#003566;'>
💉 COVID-19 Vaccine Sentiment Analysis
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;
font-size:20px;'>
Predict whether a vaccine-related tweet is
<b>Positive 😊</b>,
<b>Neutral 😐</b>,
or
<b>Negative 😔</b>
</p>
""", unsafe_allow_html=True)

st.write("")


# ===========================================
# INPUT BOX
# ===========================================

tweet = st.text_area(
    "✍ Enter Tweet",
    height=180,
    placeholder="Example: The vaccine is safe and effective."
)

st.write("")
# ===========================================
# Prediction Button
# ===========================================

if st.button("🔍 Predict Sentiment", use_container_width=True):

    if tweet.strip() == "":
        st.warning("⚠ Please enter a tweet.")

    else:

        # Clean Tweet
        cleaned = clean_text(tweet)

        # TF-IDF
        vector = tfidf.transform([cleaned])

        # Prediction
        prediction = model.predict(vector)[0]

        # Probability (if supported)
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vector).max() * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        # Adjust these labels if your dataset uses different encoding
        if prediction == 1:
            st.error("😔 Negative Sentiment")

        elif prediction == 2:
            st.info("😐 Neutral Sentiment")

        elif prediction == 3:
            st.success("😊 Positive Sentiment")

        else:
            st.write(f"Prediction: {prediction}")

        if confidence is not None:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )
