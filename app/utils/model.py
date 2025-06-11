# app/utils/model.py
import joblib

def load_model():
    return joblib.load("model/resume_model.pkl")

def load_vectorizer():
    return joblib.load("model/tfidf_vectorizer.pkl")
