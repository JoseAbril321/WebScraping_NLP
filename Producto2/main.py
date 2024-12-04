from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import re
import joblib
from pydantic import BaseModel, HttpUrl

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

with open("modelo_entrenado.joblib", "rb") as f:
    test = joblib.load(f)

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
def predict(data: InputData):
    resultados = []
    try:
        for url in data:
            variables = None
            # Extraer texto de la URL
            text = extraer_text(url.text)
            vectorized_input = vectorizer.transform([text]).toarray()

            # Predicciones del primer modelo
            output_model = model.predict(vectorized_input).tolist()[0]

            # Excluir la posición 11
            variables = [float(output_model[i]) for i in range(len(output_model)) if i != 11]

            # Predicción del segundo modelo
            import numpy as np
            variables_np = np.array([variables])  # Convertir a formato 2D
            output_test = test.predict(variables_np).tolist()

            resultados.append({
                "URL": url.text,
                "Precio URL": float(output_model[11]),
                "Precio Predicho": output_test
            })
            
        return {
            "resultados": resultados
        }
    except Exception as e:
        return {
            "error": str(e),
            "respuesta": resultados
        }

