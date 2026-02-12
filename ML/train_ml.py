#TRAINING AND SAVING THE MODEL 
#RUN ONCCE



# train_ml.py
import joblib
import pandas as pd
from best_model import detectionmodel
from train_test import split_function
from preprocessing import data_cleaning, text_join, textpreprocessor, tfidf_features, smote_method

def run_training_pipeline():
    # 1. Load Dataset
    df = pd.read_excel(r"C:\Users\aarat\Desktop\Bvoc IT\sem6\proj_s6\main\emscad_cleaned_excel.xlsx")

    # 2. Preprocessing
    df = data_cleaning(df)
    df = text_join(df)
    df['text'] = df['text'].apply(lambda x: textpreprocessor().preprocess(x))

    # 3. Split & Feature Extraction
    X_train, X_test, y_train, y_test = split_function(df)
    X_train, X_test, vectorizer = tfidf_features(X_train, X_test)

    # 4. Balance & Train
    X_train, y_train = smote_method(X_train, y_train)
    model_wrapper = detectionmodel(model_type="random_forest")
    model_wrapper.training(X_train, y_train)

    # 5. SAVE THE ASSETS (Crucial for the App)
    joblib.dump(model_wrapper.model, "detection_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Model and Vectorizer saved successfully!")

if __name__ == "__main__":
    run_training_pipeline()