import logging
import os
import time
from datetime import date

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.exceptions import MlflowException, RestException

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask("wind-prediction")

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


def load_model_with_retry(max_retries=10, delay_seconds=10):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Poging %s/%s om model '%s' te laden...", attempt, max_retries, model_uri)
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logger.info(" Model succesvol geladen in de web-service!")
            return loaded_model
        except (RestException, MlflowException, ConnectionError) as e:
            logger.warning(" Model kon (nog) niet geladen worden: %s", e)
            if attempt < max_retries:
                logger.info(
                    " Wachten voor %s seconden voordat we het opnieuw proberen...", delay_seconds
                )
                time.sleep(delay_seconds)
            else:
                logger.error(" Maximaal aantal pogingen bereikt. Web-service start ZONDER model.")
                return None
    return None


model = load_model_with_retry()


def build_features(data: dict) -> pd.DataFrame:
    d = date.fromisoformat(data["datum"])
    doy = d.timetuple().tm_yday
    m = d.month

    row = {
        "geo_windspeed_10m": data["geo_windspeed_10m"],
        "geo_windspeed_30m": data.get("geo_windspeed_30m", 0),
        "ukkel_windspeed_10m": data.get("ukkel_windspeed_10m", 0),
        "maand_sin": np.sin(2 * np.pi * m / 12),
        "maand_cos": np.cos(2 * np.pi * m / 12),
        "dag_sin": np.sin(2 * np.pi * doy / 365),
        "dag_cos": np.cos(2 * np.pi * doy / 365),
        "weekdag": d.weekday(),
    }
    return pd.DataFrame([row])[FEATURES]


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if model is None:
        return jsonify({"error": "Model is nog niet geladen of beschikbaar in MLflow"}), 503

    try:
        data = request.get_json()
        df_features = build_features(data)
        prediction = model.predict(df_features)
        result = {
            "predicted_kwh": round(float(prediction[0]), 2),
            "predicted_mwh": round(float(prediction[0]) / 1000, 3),
        }
        return jsonify(result)
    except (KeyError, ValueError, TypeError) as e:
        logger.error("Fout tijdens predictie (slechte input data): %s", e)
        return jsonify({"error": f"Ongeldige input: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
