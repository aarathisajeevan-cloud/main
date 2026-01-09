from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(df["final_text"], df["fraudulent"])

joblib.dump(pipeline, "model.joblib")
