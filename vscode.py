import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset (adjust path as needed)
df = pd.read_excel(r"C:\Users\aarat\Desktop\Bvoc IT\sem6\proj_s6\main\emscad_cleaned_excel.xlsx")

# Preprocessing functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply preprocessing
text_cols = ["description", "benefits", "city", "requirements"]
df[text_cols] = df[text_cols].fillna("unknown")
text_join = ["title","description","company_profile","requirements","benefits"]
df['text']= df[text_join].agg(' '.join,axis =1)
df["clean_text"] = df['text'].apply(clean_text)

# Tokenization, stopword removal, lemmatization
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def tokenize_text(text):
    return word_tokenize(text)

df["tokens"] = df["clean_text"].apply(tokenize_text)
df["tokens"] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
df["tokens"]=df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word)for word in tokens])

def join_tokens(tokens):
    return " ".join(tokens)
df["final_text"] = df["tokens"].apply(join_tokens)

# Split data
X = df["final_text"]
y = df["fraudulent"]
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=20000 , ngram_range=(1,2) , stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models
logreg = LogisticRegression(max_iter=1000 , random_state=42)
logreg.fit(X_train_tfidf , y_train)

mnb = MultinomialNB()
mnb.fit(X_train_tfidf,y_train)

sup_vec = LinearSVC(max_iter=5000, random_state=42, dual=False)
sup_vec.fit(X_train_tfidf,y_train)

rfc = RandomForestClassifier(n_estimators=200, random_state=42 , n_jobs=-1)
rfc.fit(X_train_tfidf,y_train)

models = {"Logistic Regression": logreg, "Naive Bayes": mnb, "Support Vector Machine": sup_vec, "Random Forest": rfc}

# Streamlit app
st.title("Fraudulent Job Posting Detector")

user_input = st.text_area("Enter job description text:")
selected_model = st.selectbox("Select Model:", list(models.keys()))

if st.button("Predict"):
    if user_input:
        # Preprocess input
        cleaned = clean_text(user_input)
        tokens = tokenize_text(cleaned)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        final_text = " ".join(tokens)
        
        # Vectorize
        input_tfidf = tfidf.transform([final_text])
        
        # Predict
        model = models[selected_model]
        prediction = model.predict(input_tfidf)[0]
        prob = model.predict_proba(input_tfidf)[0][1] if hasattr(model, 'predict_proba') else "N/A"
        
        st.write(f"Prediction: {'Fraudulent' if prediction == 1 else 'Real'}")
        if prob != "N/A":
            st.write(f"Probability of Fraud: {prob:.2f}")
    else:
        st.write("Please enter text.")