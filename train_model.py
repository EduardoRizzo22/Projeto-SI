import os
import joblib
import pandas as pd
from app.utils.nlp_processing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Carregar dados
labels = pd.read_csv("data/labels.csv")
data = []
for fname in labels['filename']:
    with open(f"data/resumes/{fname}", "r", encoding="utf-8") as f:
        data.append(f.read())

texts = [preprocess_text(d) for d in data]

# Vetorizador e modelo
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)
y = labels['label']

model = RandomForestClassifier()
model.fit(X, y)

# Salvar modelo e vetorizador
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/resume_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
