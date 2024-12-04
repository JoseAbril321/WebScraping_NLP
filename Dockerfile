FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY vectorizer.joblib .
COPY multioutput_model.joblib .
COPY modelo_entrenado.joblib .

# Configurar NLTK
RUN mkdir -p /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords wordnet

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
