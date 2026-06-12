from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Stroke Prediction API", description="API para classificar risco de AVC", version="1.0")

MODEL_PATH = 'best_model.joblib'

# Tenta carregar o modelo treinado na inicialização
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modelo carregado com sucesso!")
    else:
        model = None
        print(f"Aviso: Modelo {MODEL_PATH} não encontrado. Certifique-se de exportá-lo do notebook.")
except Exception as e:
    model = None
    print(f"Erro ao carregar o modelo: {e}")

class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.post("/predict")
def predict_stroke(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor. Por favor, exporte o modelo do notebook como 'best_model.joblib'.")
    
    # Converte a entrada para DataFrame para o pipeline do Scikit-Learn (ColumnTransformer)
    input_data = {
        "gender": [data.gender],
        "age": [data.age],
        "hypertension": [data.hypertension],
        "heart_disease": [data.heart_disease],
        "ever_married": [data.ever_married],
        "work_type": [data.work_type],
        "Residence_type": [data.Residence_type],
        "avg_glucose_level": [data.avg_glucose_level],
        "bmi": [data.bmi],
        "smoking_status": [data.smoking_status]
    }
    input_df = pd.DataFrame(input_data)
    
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "risk_status": "Alto Risco" if prediction[0] == 1 else "Baixo Risco"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao realizar predição: {str(e)}")
