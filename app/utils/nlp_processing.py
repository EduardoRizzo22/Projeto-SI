import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

STOPWORDS = set(open("data/stopwords.txt").read().splitlines())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

