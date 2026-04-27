from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

app = FastAPI(title="Credit Card Fraud Detection API")
model    = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

class TransactionInput(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is live"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: TransactionInput):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[features]

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    result = "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE"
    logging.info(f"Amount: {data.Amount} | Prediction: {result} | Confidence: {probability:.2f}")

    return {
        "prediction": result,
        "confidence": round(float(probability), 4),
        "is_fraud": bool(prediction)
    }