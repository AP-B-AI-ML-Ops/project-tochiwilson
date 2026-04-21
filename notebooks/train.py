import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def read_dataframe(filename: str):
    """Laadt de opgeschoonde dataset in."""
    df = pd.read_csv(filename)
    df["tijd"] = pd.to_datetime(df["tijd"])
    return df


def run_train():
    # 1. Laad de gecombineerde data in
    df = read_dataframe("../data/train_data_wind.csv")

    # 2. Features en Target definiëren
    features = ["geo_windspeed_10m"]
    target = "elia wind kwh"

    X = df[features]
    y = df[target].values

    # 3. Splits de data in Train (80%) en Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Start de MLflow Tracking
    with mlflow.start_run():

        # Train het model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Voorspellen op de ongeziene test-data
        predictions = lr.predict(X_test)

        # Bereken de foutmarge (RMSE)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Model succesvol getraind! Test RMSE: {rmse:.2f} kWh")

        # Log alles netjes weg naar MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", features)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_metric("rmse", rmse)

        # Sla het model op in de registry
        mlflow.sklearn.log_model(lr, "wind_energy_model")

    return lr


if __name__ == "__main__":
    run_train()
