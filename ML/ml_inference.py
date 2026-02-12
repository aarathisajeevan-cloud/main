# inference.py
import joblib
from ml_preprocessing import textpreprocessor

from fake_words import fake_words

# Load the saved assets
model = joblib.load("detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
fw = fake_words()

def predict_from_user(text):
    """
    Takes raw text, cleans it, vectorizes it, and returns a result string.
    """
    # 1. Rule-based check first (before cleaning removes numbers/symbols)
    # 1. Rule-based check first (before cleaning removes numbers/symbols)
    pay_match, pay_words = fw.has_payment_request(text)
    easy_match, easy_words = fw.suspicious_easy_job(text)

    if pay_match or easy_match:
         return {
             "status": "FAKE",
             "reason": "Rule-based",
             "words": pay_words + easy_words
         }

    # Initialize your preprocessor class
    preprocessor = textpreprocessor() 
    
    clean_text = preprocessor.preprocess(text)
    vect_text = vectorizer.transform([clean_text])
    prediction = model.predict(vect_text)

    if prediction[0] == 1:
        return {
            "status": "FAKE",
            "reason": "Model-based",
            "words": []
        }
    else:
        return {
            "status": "REAL",
            "reason": "Model-based",
            "words": []
        }