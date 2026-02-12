# # inference.py
# import joblib
# from preprocessing import textpreprocessor

# # Load the saved assets
# model = joblib.load("detection_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# def predict_from_user(text):
#     """
#     Takes raw text, cleans it, vectorizes it, and returns a result string.
#     """
#     # Initialize your preprocessor class
#     preprocessor = textpreprocessor() 
    
#     clean_text = preprocessor.preprocess(text)
#     vect_text = vectorizer.transform([clean_text])
#     prediction = model.predict(vect_text)

#     if prediction[0] == 1:
#         return " This job post looks FAKE."
#     else:
#         return " This job post looks REAL."