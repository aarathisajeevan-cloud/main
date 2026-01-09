import streamlit as st
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load trained pipeline
model = joblib.load("model.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ---------------- UI ----------------
st.set_page_config(page_title="Fake Job Posting Detection")

st.title("🕵️ Fake Job Posting Detection")

user_input = st.text_area("Enter job description:")

if st.button("Predict"):
    if user_input.strip():
        final_text = preprocess(user_input)
        prediction = model.predict([final_text])[0]

        if prediction == 1:
            st.error("🚨 This job posting is likely FAKE")
        else:
            st.success("✅ This job posting is likely REAL")
    else:
        st.warning("Please enter a job description.")
