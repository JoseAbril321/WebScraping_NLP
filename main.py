from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API funcionando correctamente"}

@app.post("/predict")
def predict(data: dict):
    # Ejemplo de predicción simulada
    prediction = {"result": "Predicción exitosa"}
    return prediction
