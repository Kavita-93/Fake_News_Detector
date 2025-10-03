import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import nltk
import joblib
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set Streamlit page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="auto"
)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Data
df_fake = pd.read_csv(r"C:/Users/DELL/Desktop/Fake_News_Detection_System/politifact_fake.csv")
df_real = pd.read_csv(r"C:/Users/DELL/Desktop/Fake_News_Detection_System/politifact_real.csv")

df_real['target'] = 0  # Real
df_fake['target'] = 1  # Fake

# Reserve 10 samples for manual testing
df_real = df_real.iloc[:-10]
df_fake = df_fake.iloc[:-10]

# Combine and shuffle
data = pd.concat([df_real, df_fake]).reset_index(drop=True)
data = data.drop(['news_url', 'tweet_ids'], axis=1, errors='ignore')
data = data.sample(frac=1).reset_index(drop=True)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['title'] = data['title'].apply(clean_text)

# Features and Labels
x = data['title']
y = data['target']

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Support Vector Machine Model
svm = SVC(kernel='linear', probability=True, random_state=0)
svm.fit(xv_train, y_train)

# Save model and vectorizer
joblib.dump(svm, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #007acc;'> Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news headline or article text below to check if it's <b>Fake</b> or <b>Real</b>.</p>", unsafe_allow_html=True)

# Load trained model & vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Optional: Add logo if available
# st.image("news_logo.png", width=100)

#Input
news_input = st.text_area("🗞 Paste your news text here:", height=150)

# Predict
if st.button("🔍 Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([clean_text(news_input)])
        prediction = model.predict(transform_input)

        if prediction[0] == 0:
            st.success(" This appears to be *Real News*.")
        else:
            st.error(" This appears to be *Fake News*.")

        # --- Feedback Section ---
        st.markdown("###  Was this prediction correct?")
        feedback = st.radio("Your feedback:", ["Yes", "No"], index=None, horizontal=True)

        if feedback:
            # Save feedback (append to file or database)
            with open("user_feedback.csv", "a", encoding="utf-8") as f:
                f.write(f'"{news_input.replace(",", " ")[:300]}",{prediction[0]},{feedback}\n')
            st.success(" Thank you for your feedback!")

    else:
        st.warning("⚠ Please enter some text to analyze.")
