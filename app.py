# Importing libraries
import streamlit as st
from model_pred import evaluate_pred


st.title("Fake News Detection System")
st.markdown("A simple web application that detects whether a news article is fake or real")
st.markdown("**HOW TO USE**: Enter the details of the news article in the 'Main' tab and click on the 'Submit' button to see the predictions in the 'Result' tab.")
st.markdown("**MODELS**: Logistic Regression, K-Nearest Neighbors, Random Forest, Feedforward Neural Network")

main, result = st.tabs(["Main", "Result"])

with main:
    st.markdown("#### Enter the details of the news article below")
    news_title: str = st.text_area("Enter the title of the news article", height=100)
    news_source: str = st.text_input("Enter the news source")
    news_retweets: int = st.number_input("Enter the number of retweets", min_value=0, max_value=500, step=1)
    submit_button = st.button("Submit", icon="🔥", key="submit", help="Submit and move to the Result tab")

if submit_button:
    evaluate_pred(main, result, news_title, news_source, news_retweets)
