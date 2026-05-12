import os
import numpy as np
import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from datetime import date

app = Flask("wind-prediction")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
model = mlflow.pyfunc.load_model("models:/wind-forecaster-best-model/1")

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
    data = request.get_json()
    df_features = build_features(data)
    prediction = model.predict(df_features)
    result = {
        "predicted_kwh": round(float(prediction[0]), 2),
        "predicted_mwh": round(float(prediction[0]) / 1000, 3),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
