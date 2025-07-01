import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Carrega lista de stopwords
STOPWORDS = set(open("data/stopwords.txt").read().splitlines())

# Função de pré-processamento
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# Criação do vetorizar (não treinado ainda)
vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text,
    stop_words=STOPWORDS,
    max_features=5000
)
