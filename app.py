import nltk
nltk.download('stopwords')
import streamlit as st
import pickle
import numpy as np
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", '', text)              # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra whitespace
    return text

# Page config
st.set_page_config(
    page_title="📰 Fake News Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

# Header banner
st.markdown("""
<div style="
    background-color: #0e1117;
    padding: 15px;
    border-left: 6px solid #1f77b4;
    border-radius: 8px;
    margin-bottom: 20px;
">
    <h1 style='color: white;'>📰 Fake News Detection System</h1>
    <p style='color: lightgray; font-size: 16px;'>
        This tool uses <strong>Machine Learning and NLP</strong> to classify news as <strong>Fake</strong> or <strong>Real</strong>.
        <br><br>🛑 <strong>Note:</strong> This model was trained on the following categories only:
        <br>
        <code>politicsNews</code>, <code>worldnews</code>, <code>News</code>, <code>politics</code>, <code>left-news</code>, <code>Government News</code>, <code>US_News</code>, <code>Middle-east</code>
        <br><br>📢 For best results, input news related to these topics only.
    </p>
</div>
""", unsafe_allow_html=True)

# Input form
with st.form(key='newsForm'):
    user_input = st.text_area("🔎 Enter the news text here:", height=200)
    submit = st.form_submit_button(label='Predict')

# Prediction
if submit:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Show input as styled card
        st.markdown("### 📝 News Preview")
        st.markdown(f"""
        <div style="
            background-color: #262730;
            padding: 15px;
            border-left: 5px solid #00c0ff;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            line-height: 1.5;
        ">
            {user_input}
        </div>
        """, unsafe_allow_html=True)

        # Prediction logic
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        confidence = round(max(prob) * 100, 2)

        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        color = "#1f8a70" if prediction == 1 else "#8a1f1f"

        # Styled prediction box
        st.markdown(f"""
        <div style="
            background-color: {color}20;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
            color: white;
            font-size: 20px;
            text-align: center;
            border: 2px solid {color};
        ">
            ✅ Prediction: <b>{label}</b><br>
            Confidence: <b>{confidence}%</b>
        </div>
        """, unsafe_allow_html=True)

        # Color-coded progress bar
        progress_color = "green" if confidence >= 80 else "orange" if confidence < 60 else "red"
        st.markdown(f"""
        <div style="margin-top:10px; margin-bottom:15px;">
            <div style="height: 25px; background-color: lightgray; border-radius: 10px;">
                <div style="
                    width: {confidence}%;
                    height: 100%;
                    background-color: {progress_color};
                    border-radius: 10px;
                    text-align: center;
                    line-height: 25px;
                    color: white;
                    font-weight: bold;">
                    {confidence}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence feedback
        if confidence < 60:
            st.warning("⚠️ Low confidence — this result may not be reliable.")
        elif confidence < 80:
            st.info("⚠️ Medium confidence — could be a borderline case.")
        else:
            st.success("✅ High confidence — prediction is likely reliable.")

# Explanation section
with st.expander("🧠 How it works"):
    st.markdown("""
    - Preprocessing using TF-IDF Vectorization  
    - Classification with Logistic Regression  
    - Trained on ~50,000 political and world news articles (fake + real)  
    - Confidence is based on model probability
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-style: italic; color: gray; font-size: 15px; margin-top: 30px;'>
© 2025 <span style="font-family: cursive;">Shiv</span>. All rights reserved.
</div>
""", unsafe_allow_html=True)
