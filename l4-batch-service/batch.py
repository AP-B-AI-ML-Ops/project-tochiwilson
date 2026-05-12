import os
from datetime import date, datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from sqlalchemy_utils import create_database, database_exists

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "wind-forecaster-best-model"
MODEL_VERSION = "1"
RMSE_THRESHOLD = 150000  # kWh — onze huidige RMSE is ~106k dus dit is veilig

DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@database/metrics"

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


def ensure_database_exists(uri):
    if not database_exists(uri):
        create_database(uri)


def save_metrics_to_db(metrics_df: pd.DataFrame):
    ensure_database_exists(DB_URI)
    engine = create_engine(DB_URI)
    metrics_df.to_sql("evidently_metrics", engine, if_exists="append", index=False)
    print("💾 Metrics opgeslagen in database")


@task(log_prints=True)
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    print(f"✅ Model geladen: {MODEL_NAME} v{MODEL_VERSION}")
    return model


@task(log_prints=True)
def fetch_weather_forecast(wind_path: str, prod_path: str) -> pd.DataFrame:
    df_wind = pd.read_csv(wind_path)
    df_wind["date"] = pd.to_datetime(df_wind["date"])
    df_wind["dag"] = df_wind["date"].dt.normalize()

    # Gebruik de laatste 7 dagen die overlappen met de productiedata
    df_prod = pd.read_csv(prod_path)
    df_prod["tijd"] = pd.to_datetime(df_prod["tijd"]).dt.tz_localize(None)
    prod_max = df_prod["tijd"].max().normalize()

    cutoff = prod_max - timedelta(days=7)
    forecast = df_wind[(df_wind["dag"] >= cutoff) & (df_wind["dag"] <= prod_max)].copy()

    print(f"📡 Weerforecast opgehaald: {len(forecast)} dagen (tot {prod_max.date()})")
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
    merged = merged.dropna(subset=["actual_kwh", "predicted_kwh"])
    print(f"✅ Actuals gevonden voor {len(merged)} van {len(predictions)} dagen")
    return merged


@task(log_prints=True)
def compute_and_log_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        print("⚠️  Geen overlap tussen voorspellingen en actuals")
        return {}

    rmse = float(np.sqrt(mean_squared_error(df["actual_kwh"], df["predicted_kwh"])))
    mae = float(np.mean(np.abs(df["actual_kwh"] - df["predicted_kwh"])))
    mape = float(np.mean(np.abs((df["actual_kwh"] - df["predicted_kwh"]) / df["actual_kwh"])) * 100)

    print("\n📊 Batch metrics:")
    print(f"   RMSE: {rmse:,.0f} kWh")
    print(f"   MAE:  {mae:,.0f} kWh")
    print(f"   MAPE: {mape:.1f}%")

    # Evidently rapport
    df_evidently = df.rename(columns={"actual_kwh": "target", "predicted_kwh": "prediction"})
    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=df_evidently,
        current_data=df_evidently,
    )
    os.makedirs("/data/batch_results", exist_ok=True)
    report.save_html(f"/data/batch_results/evidently_report_{date.today()}.html")

    # Extraheer metrics
    report_dict = report.as_dict()
    result_data = []
    run_time = datetime.utcnow()
    for metric in report_dict.get("metrics", []):
        result_data.append(
            {
                "run_time": run_time,
                "metric_name": str(metric.get("metric", "")),
                "value": str(metric.get("result", "")),
                "model_version": MODEL_VERSION,
                "batch_date": date.today().isoformat(),
            }
        )

    # Voeg ook onze eigen metrics toe
    for metric_name, value in [("rmse", rmse), ("mae", mae), ("mape", mape)]:
        result_data.append(
            {
                "run_time": run_time,
                "metric_name": metric_name,
                "value": str(value),
                "model_version": MODEL_VERSION,
                "batch_date": date.today().isoformat(),
            }
        )

    metrics_df = pd.DataFrame(result_data)
    save_metrics_to_db(metrics_df)

    # Log naar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("batch-monitoring")
    with mlflow.start_run():
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape})
        mlflow.log_param("model_version", MODEL_VERSION)
        mlflow.log_param("batch_date", date.today().isoformat())

    # Prefect artifact
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

## Retraining drempel: {RMSE_THRESHOLD:,.0f} kWh
## Status: {"🔴 RETRAINING NODIG" if rmse > RMSE_THRESHOLD else "🟢 OK"}
""",
    )
    return {"rmse": rmse, "mae": mae, "mape": mape}


@task(log_prints=True)
def check_retraining_needed(metrics: dict) -> bool:
    if not metrics:
        return False
    rmse = metrics.get("rmse", 0)
    if rmse > RMSE_THRESHOLD:
        print(f"🔴 RMSE {rmse:,.0f} > drempel {RMSE_THRESHOLD:,.0f} — retraining triggeren!")
        return True
    print(f"🟢 RMSE {rmse:,.0f} onder drempel {RMSE_THRESHOLD:,.0f} — geen retraining nodig")
    return False


@task(log_prints=True)
def trigger_retraining():
    """Triggert de training pipeline opnieuw via Prefect."""
    import subprocess

    print("🔄 Retraining gestart...")
    subprocess.run(["python", "/app/retrain.py"], check=True)


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
    forecast_df = fetch_weather_forecast(wind_path, prod_path)
    features_df = build_features(forecast_df)
    predictions = run_inference(model, features_df)
    merged = fetch_actuals(prod_path, predictions)
    metrics = compute_and_log_metrics(merged)
    save_predictions(merged, output_path)
    needs_retraining = check_retraining_needed(metrics)
    if needs_retraining:
        trigger_retraining()


if __name__ == "__main__":
    batch_flow()
