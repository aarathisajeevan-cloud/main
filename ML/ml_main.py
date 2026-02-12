#run once 

import joblib
from preprocessing import textpreprocessor
from fake_words import fake_words

# Load the saved assets
try:
    model = joblib.load("detection_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit()

def predict_from_user(text):
    preprocessor = textpreprocessor() 
    clean_text = preprocessor.preprocess(text)
    print(f"Original Text: {text}")
    print(f"Cleaned Text: {clean_text}")
    
    vect_text = vectorizer.transform([clean_text])
    prediction = model.predict(vect_text)

    if prediction[0] == 1:
        return "FAKE"
    else:
        return "REAL"

text = "no experience needed for joining but needed to know more about coding and also pay 5000 for regestration₹400"

print(f"Model Prediction: {predict_from_user(text)}")

# Check fake_words logic
fw = fake_words()
print(f"fake_words.has_payment_request: {fw.has_payment_request(text)}")
print(f"fake_words.suspicious_easy_job: {fw.suspicious_easy_job(text)}")
