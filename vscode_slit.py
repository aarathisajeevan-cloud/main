import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report, precision_score,recall_score, roc_auc_score, confusion_matrix,f1_score
import warnings
warnings.filterwarnings("ignore")

#page configuration
st.set_page_config(page_title="Fake Job Posting detection",page_icon=r"C:\Users\aarat\Downloads\search_icon.png",layout="wide", initial_sidebar_state="expanded")
#css for better syling
st.markdown("""<style>
            .main{
            padding-top:2rem;
            }
            .stTabs[data-baseweb="tab-list"]button[data-testid = "stMarkdownContainer"]p{font-size:1.2rem;
            }
            .metric-box {
                backgroundcolor:#f0f2f6;
                padding :1rem
                border-radius:0.5rem;
                margin:0.5rem 0;
            }
            </style>""",unsafe_allow_html=True
            )
#dwnld nltk resources
st.cache_resource
def nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

nltk_resources()

#text preprocessing class
