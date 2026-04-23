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

app = FastAPI(title="Stock Movement Predictor API")
model    = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

class StockInput(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    SMA_10: float
    SMA_30: float
    RSI: float
    MACD: float
    BB_high: float
    BB_low: float

@app.get("/")
def root():
    return {"message": "Stock Movement Predictor API is live"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: StockInput):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    direction = "UP 📈" if prediction == 1 else "DOWN 📉"

    logging.info(f"Input: {data.dict()} | Prediction: {direction} | Confidence: {probability:.2f}")

    return {
        "prediction": direction,
        "confidence": round(float(probability), 4),
        "signal": int(prediction)
    }