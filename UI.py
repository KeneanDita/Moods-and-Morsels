import streamlit as st
import joblib
import numpy as np

# Load the trained model + vectorizer
model = joblib.load("sentiment_model.pkl")  # replace with your model filename
vectorizer = joblib.load("vectorizer.pkl")  # replace with your vectorizer filename

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis App")
st.write(
    "Enter a sentence below and I’ll predict whether it’s **Positive**, **Negative**, or **Neutral**."
)

# User input
user_input = st.text_area("✍️ Type a sentence:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Transform input
        X = vectorizer.transform([user_input])

        # Get probability for each class
        probs = model.predict_proba(X)[0]
        labels = model.classes_  # should be ["negative", "positive"]

        # Assign probabilities
        prob_dict = dict(zip(labels, probs))
        pos_prob = prob_dict.get("positive", 0)
        neg_prob = prob_dict.get("negative", 0)

        # Define neutral threshold
        if abs(pos_prob - neg_prob) < 0.2:  # tweak this margin
            sentiment = "Neutral 😐"
        else:
            sentiment = "Positive 😀" if pos_prob > neg_prob else "Negative 😞"

        # Show results
        st.subheader("Prediction:")
        st.success(sentiment)

        st.write("**Confidence:**")
        st.write(f"👍 Positive: {pos_prob:.2f}")
        st.write(f"👎 Negative: {neg_prob:.2f}")
    else:
        st.warning("Please enter a sentence before analyzing.")
