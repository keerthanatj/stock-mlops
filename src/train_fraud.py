import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data():
    df = pd.read_csv("data/fraud_data.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, preds, zero_division=0), 4),
    }

def train_and_log(name, model, params, X_train, X_test, y_train, y_test):
    print(f"\n Training: {name}...")
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_name", name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, f"{name}_model",
            registered_model_name="FraudDetector"
        )

        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1 Score:  {metrics['f1_score']}")

    return model, metrics

def train():
    mlflow.set_experiment("credit-card-fraud-detection")
    X_train, X_test, y_train, y_test, features = load_data()

    models = [
        (
            "GradientBoosting",
            Pipeline([("scaler", StandardScaler()),
                      ("model", GradientBoostingClassifier(
                          n_estimators=200, learning_rate=0.05,
                          max_depth=5, random_state=42))]),
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5}
        ),
        (
            "RandomForest",
            Pipeline([("scaler", StandardScaler()),
                      ("model", RandomForestClassifier(
                          n_estimators=200, max_depth=8,
                          random_state=42))]),
            {"n_estimators": 200, "max_depth": 8}
        ),
        (
            "LogisticRegression",
            Pipeline([("scaler", StandardScaler()),
                      ("model", LogisticRegression(
                          max_iter=1000, C=0.1, random_state=42))]),
            {"max_iter": 1000, "C": 0.1}
        ),
        (
            "SVM",
            Pipeline([("scaler", StandardScaler()),
                      ("model", SVC(
                          kernel="rbf", C=10.0,
                          probability=True, random_state=42))]),
            {"kernel": "rbf", "C": 10.0}
        ),
    ]

    results = {}
    best_model = None
    best_f1 = -1
    best_name = ""

    for name, model, params in models:
        trained_model, metrics = train_and_log(
            name, model, params, X_train, X_test, y_train, y_test
        )
        results[name] = metrics
        if metrics["f1_score"] > best_f1:
            best_f1    = metrics["f1_score"]
            best_model = trained_model
            best_name  = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(features,   "models/features.pkl")

    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-"*60)
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"{name:<22} {m['accuracy']:>9} {m['precision']:>10} "
              f"{m['recall']:>8} {m['f1_score']:>8}{marker}")
    print("="*60)
    print(f"\n Best model saved: {best_name} (F1 = {best_f1})")

if __name__ == "__main__":
    train()