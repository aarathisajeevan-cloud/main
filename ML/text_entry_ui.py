# app.py
import streamlit as st
from ml_inference import predict_from_user

def result_text():
    
    st.subheader("Job Description Authenticator")
    st.write("Paste the job description below to check if it's potentially fraudulent.")

    user_text = st.text_area(
        "Enter job description",
        placeholder="Paste job details here...",
        height=250
    )

    result = None
    if st.button("Analyze Job Post"):
        if user_text.strip():
            with st.spinner('Analyzing patterns...'):
                result = predict_from_user(user_text)
                
                if result["status"] == "FAKE":
                    st.error(f"🚨 Warning: This job post looks FAKE.")
                    if result["words"]:
                        st.warning(f"Suspicious words found: {', '.join(result['words'])}")
                else:
                    st.success(f"✅ Verified: This job post looks REAL.")
        else:
            st.warning("Please enter some text first!")

    return result


if __name__ == "__main__":
    st.set_page_config(page_title="Fake Job Detector", layout="wide")
    result_text()