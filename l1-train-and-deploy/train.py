import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

FEATURES = [
    "geo_windspeed_10m",
    "geo_windspeed_30m",
    "ukkel_windspeed_10m",
    "maand_sin",
    "maand_cos",
    "dag_sin",
    "dag_cos",
    "weekdag",
]
TARGET = "elia_wind_kwh_gemiddeld"


def read_dataframe(filename: str) -> pd.DataFrame:
    """Laadt de opgeschoonde dataset in."""
    df = pd.read_csv(filename)
    df["dag"] = pd.to_datetime(df["dag"])
    return df


def run_train(data_path: str = "../data/train_data_wind.csv") -> LinearRegression:
    # Configureer MLflow Tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("wind-energy-baseline")

    # 1. Laad de gecombineerde data in
    df = read_dataframe(data_path)

    # 2. Features en Target — gebruik alleen kolommen die ook echt aanwezig zijn
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].fillna(df[available_features].median())
    y = df[TARGET].values

    # 3. Splits de data in Train (80%) en Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Start de MLflow Tracking
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        predictions = lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Model succesvol getraind! Test RMSE: {rmse:.2f} kWh")

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", available_features)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, artifact_path="wind_energy_model")

    return lr


if __name__ == "__main__":
    run_train()
