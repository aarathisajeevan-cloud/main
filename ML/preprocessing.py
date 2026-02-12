#preprocessing, cleaning, feature engineering

#NLTK LIBRARIES AND RESOURCES
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#importing the libraries and Models

import pandas as pd #numeric calculations
import numpy as np #numeric calculations
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization
import re #regular expressions for text cleaning

#scikit-learn models and functions for ML training
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
# #warnings supperssion
import warnings
warnings.filterwarnings("ignore")



#loading dataset
df = pd.read_excel(r"C:\Users\aarat\Desktop\Bvoc IT\sem6\proj_s6\main\emscad_cleaned_excel.xlsx")



#DATA CLEANING & PREPROCESSING
def data_cleaning(df):
    #print("Analysing Dataset : ")
    #df.info()
    #df.head()
    # print("shape:",df.shape)
    # print("columns:",df.columns)
    # print(df)

    # print(f"Count of Fraudulent values : {df['fraudulent'].value_counts()}")
    # print(f"Percentage of Fraudulent value counts{df['fraudulent'].value_counts(normalize=True)*100}")

    # print(f"Null values in the dataset : {df.isnull().sum()}")

    # print(f"Sum of Null Values in the dataset {df.isnull().sum().sum()}")
    return df


def text_join(df):
    df = df.copy()
    cols =  ["title","description","company_profile","requirements","benefits"]
    df['text'] = df[cols].fillna("unknown").agg(" ".join,axis=1)
    return df

df =  text_join(df)
# print(df['text'].head(10))

class textpreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self,text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_text(self,text):
        return word_tokenize(text)

    def remove_stopwords(self,tokens):
        return [word for word in tokens if word not in self.stopwords]

    def lemmatize(self,tokens):
        return[self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess(self,text):
        text = self.clean_text(text)
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)
    
tp = textpreprocessor()#initialized preprocessor
#print(df.columns)

#FEATURE EXTRACTION/ENGNEERING
from train_test import split_function
X_train,X_test,y_train,y_test = split_function(df,text_cols="text",target_col="fraudulent")

def tfidf_features(X_train,X_test,max_features=3000,ngram_range=(1,2)):
    tfidf = TfidfVectorizer(max_features=max_features , ngram_range=ngram_range , stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(X_train_tfidf.shape)
    print(X_test_tfidf.shape)

    return X_train_tfidf,X_test_tfidf,tfidf

#balancing the dataset using smote method

def smote_method(X_train , y_train,random_state = 42):
    smote = SMOTE(random_state=random_state) 
    X_resampled,y_resampled = smote.fit_resample(X_train,y_train)
    return X_resampled,y_resampled




