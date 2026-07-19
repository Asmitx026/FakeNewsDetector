import streamlit as st
import pandas as pd
import joblib

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def evaluate_pred(main, result, sentence: str, source: str, retweets: int) -> None:
    with main:
        st.success("Switch to Result tab. Processing the input data...")
    # Load the pre-trained models
    models = {
        "Logistic Regression": joblib.load("models/lr_model.pkl"),
        "K-Nearest Neighbors": joblib.load("models/knn_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "Feedforward Neural Network": joblib.load("models/fnn_model.pkl")
    }

    # Load the preprocessor
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'title': [sentence],
        'source_domain': [source],
        'tweet_num': [retweets]
    })

    with result:
        st.write("#### Input Data:")
        st.dataframe(input_data)

    # Data Preprocessing
    lemmatizer = WordNetLemmatizer()

    sentence:str = re.sub(r'[^\w\s]', '', sentence) # removes punctuations and special characters
    tokens: list[str] = word_tokenize(sentence.lower())
    lemmatized_tokens: list[str] = [lemmatizer.lemmatize(token) for token in tokens]
    for word in lemmatized_tokens: # removes stopwords
        if word in stopwords.words('english'):
            lemmatized_tokens.remove(word)
    sentence: str = ' '.join(lemmatized_tokens)
    input_data['title'] = [sentence]
    input_data['source_domain'] = input_data['source_domain'].str.replace('www.', '', regex=False).str.rsplit(".").str[0]

    for name, model in models.items():
        processed_data = preprocessor.transform(input_data)
        if name == "Feedforward Neural Network":
            prediction = (model.predict(processed_data) > 0.5).astype(int)
            probability = model.predict(processed_data)[0][0]
        else:
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)[0][1]

        
        with result:
            st.markdown(f"### **{name}** Prediction")
            if prediction[0] == 1:
                st.markdown(f"Prediction Probability: {probability * 100:.2f}%")
                st.success("The news article is likely to be REAL.")
            else:
                st.markdown(f"Prediction Probability: {probability * 100:.2f}%")
                st.error("The news article is likely to be FAKE.")
            
