import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

logistic_classifier = pickle.load(open(r'E:\PYTHONCLASSVSCODE\Streamlit\logistic_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open(r'E:\PYTHONCLASSVSCODE\Streamlit\tfidf_vectorizer.pkl', 'rb'))

corpus = []
review = ""

st.title("FeedBack Analysis App")
inputText = st.text_area("Enter your review here:🙂🙂")

if inputText:
    review = re.sub('[^a-zA-Z]', ' ', inputText)
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)
    corpus = np.array(corpus)

if st.button("Model Prediction", type='primary'):
    if inputText:
        review_vector = tfidf_vectorizer.transform(corpus)
        predict = logistic_classifier.predict(review_vector)
        if predict[0] == 1:
            st.success(f"Your review is positive.")
        else:
            st.warning(f"Your review is negative.")
    else:
        st.subheader(f"Please enter your review😐😐")