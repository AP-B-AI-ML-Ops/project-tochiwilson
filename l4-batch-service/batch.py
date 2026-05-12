import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error

import mlflow
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "wind-forecaster-best-model"
MODEL_VERSION = "1"

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


@task(log_prints=True)
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    print(f"✅ Model geladen: {MODEL_NAME} v{MODEL_VERSION}")
    return model


@task(log_prints=True)
def fetch_weather_forecast(wind_path: str) -> pd.DataFrame:
    df = pd.read_csv(wind_path)
    df["date"] = pd.to_datetime(df["date"])
    df["dag"] = df["date"].dt.normalize()

    # Laatste 7 dagen als simulatie van forecast
    cutoff = df["date"].max() - timedelta(days=7)
    forecast = df[df["date"] > cutoff].copy()
    print(f"📡 Weerforecast opgehaald: {len(forecast)} dagen")
    return forecast


@task(log_prints=True)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["maand"] = df["dag"].dt.month
    df["dag_vd_jaar"] = df["dag"].dt.dayofyear
    df["weekdag"] = df["dag"].dt.weekday
    df["maand_sin"] = np.sin(2 * np.pi * df["maand"] / 12)
    df["maand_cos"] = np.cos(2 * np.pi * df["maand"] / 12)
    df["dag_sin"] = np.sin(2 * np.pi * df["dag_vd_jaar"] / 365)
    df["dag_cos"] = np.cos(2 * np.pi * df["dag_vd_jaar"] / 365)

    for col in ["geo_windspeed_30m", "ukkel_windspeed_10m"]:
        if col not in df.columns:
            df[col] = 0
    df[["geo_windspeed_30m", "ukkel_windspeed_10m"]] = df[
        ["geo_windspeed_30m", "ukkel_windspeed_10m"]
    ].fillna(0)
    return df


@task(log_prints=True)
def run_inference(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    predictions = model.predict(X)
    results = df[["dag"]].copy()
    results["predicted_kwh"] = predictions
    results["predicted_mwh"] = predictions / 1000
    print(f"🔮 Voorspellingen gemaakt voor {len(results)} dagen")
    return results


@task(log_prints=True)
def fetch_actuals(prod_path: str, predictions: pd.DataFrame) -> pd.DataFrame:
    prod = pd.read_csv(prod_path)
    prod["tijd"] = pd.to_datetime(prod["tijd"]).dt.tz_localize(None)
    prod["dag"] = prod["tijd"].dt.normalize()

    actuals = prod.groupby("dag").agg(actual_kwh=("elia wind kwh", "mean")).reset_index()

    merged = pd.merge(predictions, actuals, on="dag", how="inner")
    print(f"✅ Actuals gevonden voor {len(merged)} van {len(predictions)} dagen")
    return merged


@task(log_prints=True)
def compute_and_log_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        print("⚠️  Geen overlap tussen voorspellingen en actuals")
        return {}

    rmse = np.sqrt(mean_squared_error(df["actual_kwh"], df["predicted_kwh"]))
    mae = float(np.mean(np.abs(df["actual_kwh"] - df["predicted_kwh"])))
    mape = float(np.mean(np.abs((df["actual_kwh"] - df["predicted_kwh"]) / df["actual_kwh"])) * 100)

    metrics = {"rmse": rmse, "mae": mae, "mape": mape, "n_days": len(df)}

    print("\n📊 Batch metrics:")
    print(f"   RMSE: {rmse:,.0f} kWh")
    print(f"   MAE:  {mae:,.0f} kWh")
    print(f"   MAPE: {mape:.1f}%")

    # Log naar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("batch-monitoring")
    with mlflow.start_run():
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape})
        mlflow.log_param("model_version", MODEL_VERSION)
        mlflow.log_param("batch_date", date.today().isoformat())
        mlflow.log_param("n_days", len(df))

    # Prefect artifact (zichtbaar in UI)
    create_markdown_artifact(
        key="batch-metrics-report",
        markdown=f"""# Batch Monitoring Report

## Model: {MODEL_NAME} v{MODEL_VERSION}
## Datum: {date.today()}

| Metric | Waarde |
|:-------|-------:|
| RMSE   | {rmse:,.0f} kWh |
| MAE    | {mae:,.0f} kWh |
| MAPE   | {mape:.1f}% |
| Dagen  | {len(df)} |
""",
    )
    return metrics


@task(log_prints=True)
def save_predictions(df: pd.DataFrame, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    pred_file = os.path.join(output_path, f"predictions_{date.today()}.csv")
    df.to_csv(pred_file, index=False)
    print(f"💾 Opgeslagen: {pred_file}")


@flow(name="Wind Energy Batch Service", log_prints=True)
def batch_flow(
    wind_path: str = "/data/wind_final.csv",
    prod_path: str = "/data/productie_combined.csv",
    output_path: str = "/data/batch_results",
):
    model = load_model()
    forecast_df = fetch_weather_forecast(wind_path)
    features_df = build_features(forecast_df)
    predictions = run_inference(model, features_df)
    merged = fetch_actuals(prod_path, predictions)
    compute_and_log_metrics(merged)
    save_predictions(merged, output_path)


if __name__ == "__main__":
    batch_flow()
