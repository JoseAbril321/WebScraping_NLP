from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import re
import joblib
from pydantic import BaseModel, HttpUrl
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Cargar modelos y vectorizador
with open("vectorizer.joblib", "rb") as f:
    vectorizer = joblib.load(f)

with open("multioutput_model.joblib", "rb") as f:
    model = joblib.load(f)

app = FastAPI()

def extraer_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()

        clean_text = re.sub(r'\s+', ' ', text)  # Elimina espacios múltiples
        clean_text = re.sub(r'[^\w\s]', '', clean_text)  # Elimina puntuación
        
        tokens = word_tokenize(clean_text)
        
        lemmatized_tokens = [
            lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words
        ]
        
        return " ".join(lemmatized_tokens)
    except Exception as e:
        print(f"Error con la URL {url}: {e}")
        return ""
        
class InputData(BaseModel):
    text: HttpUrl 
@app.post("/predict")
def predict(data: List[InputData]):
    resultados = []
    try:
        for url in data:
            # Extraer texto de la URL
            text = extraer_text(url.text)
            vectorized_input = vectorizer.transform([text]).toarray()

            # Predicciones del primer modelo
            output_model = model.predict(vectorized_input).tolist()[0]
            # Agregar el resultado a la lista
            resultados.append({
                "URL": url.text,
                "variables": output_model
            })
        return {
            "resultados": resultados
        }
    except Exception as e:
        return {
            "error": str(e)
        }

