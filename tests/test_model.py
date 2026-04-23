import joblib
import pandas as pd

SAMPLE = {
    'Open': 1500.0, 'High': 1520.0, 'Low': 1490.0,
    'Close': 1510.0, 'Volume': 5000000.0,
    'SMA_10': 1505.0, 'SMA_30': 1495.0, 'RSI': 55.0,
    'MACD': 3.2, 'BB_high': 1540.0, 'BB_low': 1470.0
}

def test_model_loads():
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_prediction_is_valid():
    model    = joblib.load("models/model.pkl")
    features = joblib.load("models/features.pkl")
    df = pd.DataFrame([SAMPLE])[features]
    pred = model.predict(df)[0]
    assert pred in [0, 1]

def test_confidence_range():
    model    = joblib.load("models/model.pkl")
    features = joblib.load("models/features.pkl")
    df = pd.DataFrame([SAMPLE])[features]
    proba = model.predict_proba(df)[0]
    assert 0.0 <= max(proba) <= 1.0