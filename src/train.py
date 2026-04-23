import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

def train():
    df = pd.read_csv("data/stock_data.csv", index_col=0)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_10', 'SMA_30', 'RSI', 'MACD', 'BB_high', 'BB_low']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # shuffle=False keeps time order
    )
    mlflow.set_tracking_uri("mlruns")

    mlflow.set_experiment("stock-movement-predictor")

    with mlflow.start_run():
        # Model parameters
        params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc       = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall    = recall_score(y_test, preds)
        
            
        # Log everything to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("ticker", "INFY.NS")
        mlflow.log_param("features_count", len(features))

        # Model versioning via registry
        mlflow.sklearn.log_model(
            model, "stock_model",
            registered_model_name="StockMovementPredictor"
        )

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

        joblib.dump(model, "models/model.pkl")
        joblib.dump(features, "models/features.pkl")

if __name__ == "__main__":
    train()