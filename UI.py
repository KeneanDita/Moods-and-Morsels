import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model + tokenizer
model_path = "Models"  # folder containing config.json & pytorch_model.bin
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis with RoBERTa")
st.write(
    "Enter a sentence and I’ll classify it as **Positive**, **Negative**, or **Neutral**."
)

# User input
user_input = st.text_area("✍️ Type a sentence:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Encode input
        inputs = tokenizer(
            user_input, return_tensors="pt", truncation=True, padding=True
        )

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        # If your model has 2 labels: Negative / Positive
        if probs.shape[0] == 2:
            neg_prob, pos_prob = probs.tolist()

            # Define neutral threshold
            if abs(pos_prob - neg_prob) < 0.2:
                sentiment = "Neutral 😐"
            else:
                sentiment = "Positive 😀" if pos_prob > neg_prob else "Negative 😞"

            st.subheader("Prediction:")
            st.success(sentiment)

            st.write("**Confidence:**")
            st.write(f"👍 Positive: {pos_prob:.2f}")
            st.write(f"👎 Negative: {neg_prob:.2f}")

        # If your model has 3 labels: Negative / Neutral / Positive
        elif probs.shape[0] == 3:
            labels = ["Negative 😞", "Neutral 😐", "Positive 😀"]
            pred_idx = torch.argmax(probs).item()
            sentiment = labels[pred_idx]

            st.subheader("Prediction:")
            st.success(sentiment)

            st.write("**Confidence:**")
            for label, prob in zip(labels, probs.tolist()):
                st.write(f"{label}: {prob:.2f}")

    else:
        st.warning("Please enter a sentence before analyzing.")
