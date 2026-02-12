# app.py
import streamlit as st
from ml_main import predict_from_user

st.set_page_config(page_title="Fake Job Detector", layout="wide")

st.title("Job Description Authenticator")
st.write("Paste the job description below to check if it's potentially fraudulent.")

user_text = st.text_area(
    "Enter job description",
    placeholder="Paste job details here...",
    height=250
)

if st.button("Analyze Job Post"):
    if user_text.strip():
        with st.spinner('Analyzing patterns...'):
            result = predict_from_user(user_text)
            
            if "FAKE" in result:
                st.error(result)
            else:
                st.success(result)
    else:
        st.warning("Please enter some text first!")