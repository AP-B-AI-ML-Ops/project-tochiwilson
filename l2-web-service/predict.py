import mlflow
import pandas as pd
from flask import Flask, request, jsonify

app = Flask("wind-prediction")

# Verbind met de MLflow container (let op de docker netwerk naam)
# mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_tracking_uri("http://localhost:5000")

# Laad versie 1 van je zojuist geregistreerde Random Forest model
model = mlflow.pyfunc.load_model("models:/wind-forecaster-best-model/1")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # 1. Haal de JSON data op die de gebruiker verstuurt
    forecast = request.get_json()

    # 2. Zet de input om naar een Pandas DataFrame
    # Het model is namelijk getraind op een DataFrame met deze specifieke kolom
    features = {"geo_windspeed_10m": [forecast["geo_windspeed_10m"]]}
    df_features = pd.DataFrame(features)

    # 3. Voorspel de opbrengst
    predictions = model.predict(df_features)
    current_prediction = float(predictions[0])

    # 4. Format het antwoord
    result = {"predicted_production_kwh": current_prediction}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
