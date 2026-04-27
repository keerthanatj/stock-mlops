import joblib
import pandas as pd

def get_sample():
    df = pd.read_csv("data/fraud_data.csv")
    features = joblib.load("models/features.pkl")
    return df[features].iloc[0:1]

def test_model_loads():
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_prediction_is_valid():
    model    = joblib.load("models/model.pkl")
    sample   = get_sample()
    pred     = model.predict(sample)[0]
    assert pred in [0, 1]

def test_confidence_range():
    model  = joblib.load("models/model.pkl")
    sample = get_sample()
    proba  = model.predict_proba(sample)[0]
    assert 0.0 <= max(proba) <= 1.0

def test_accuracy_above_threshold():
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    model    = joblib.load("models/model.pkl")
    features = joblib.load("models/features.pkl")
    df       = pd.read_csv("data/fraud_data.csv")
    X        = df[features]
    y        = df["Class"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.90, f"Accuracy too low: {acc}"