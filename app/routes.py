from flask import Blueprint, render_template, request
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.nlp_processing import preprocess_text, vectorizer
from app.utils.model import load_model,load_vectorizer
import os

main = Blueprint('main', __name__)

model = load_model()
vectorizer = load_vectorizer()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/classify', methods=['POST'])
def classify():
    file = request.files['resume']
    if not file:
        return "No file uploaded", 400
    text = extract_text_from_pdf(file)
    clean_text = preprocess_text(text)
    features = vectorizer.transform([clean_text])
    prediction = model.predict(features)[0]
    return render_template('results.html', prediction=prediction)
