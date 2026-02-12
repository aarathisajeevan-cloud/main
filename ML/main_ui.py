"""Main UI for check the job post is real / fake"""


import streamlit as st
import url_entry_ui
import text_entry_ui
st.set_page_config(page_title = "Job analyser", layout="centered")

st.title("Smart Fraudulent Job Posting Analyser")
st.markdown("This project is a powerful application of Natural Language Processing (NLP) and Machine Learning designed to tackle the growing issue of deceptive recruitment. By analyzing the linguistic patterns and metadata of job advertisements, the system can distinguish between legitimate opportunities and potential scams.")

st.write("Click on the dropdown menu to select the type of input you want... ")
selection = st.selectbox("Select ",("none","Text Entry","URL Entry"))

if selection == "none":
    st.write("Please select an option from the dropdown menu")
elif selection == "Text Entry":
    st.write("Enter Job Description : ")
    text_entry_ui.result_text()
elif selection == "URL Entry":
    st.write("Paste the URL of Job Post from Indeed/LinkedIn/Naukri")
    url_entry_ui.result_url()
else:
    st.write("Please select an option")


