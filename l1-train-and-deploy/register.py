import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"

# max_features is categorisch (string), die mag niet via int() worden omgezet
RF_PARAMS_INT = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]

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


def train_and_log_model(params: dict, data_path: str):
    # Configureer MLflow Tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # Laad en splits data exact zoals in hpo.py
    df = pd.read_csv(os.path.join(data_path, "train_data_wind.csv"))

    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].fillna(df[available_features].median())
    y = df[TARGET].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # Integer params casten (MLflow slaat alles op als string)
        for param in RF_PARAMS_INT:
            if param in params:
                params[param] = int(params[param])

        # max_features: "None" (string) terug naar Python None
        if "max_features" in params and params["max_features"] == "None":
            params["max_features"] = None

            # Verwijder niet-RF params die vanuit MLflow meekomen
        rf_params = {k: v for k, v in params.items() if k not in ("features",)}

        mlflow.log_params(rf_params)
        mlflow.log_param("features", available_features)

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("test_rmse", rmse)
        print(f"  → test RMSE: {rmse:.2f} kWh")

        mlflow.sklearn.log_model(rf, artifact_path="model")


def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Haal de top N runs op uit de hyperparameter optimalisatie
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )

    print(f"Top {len(runs)} runs opgehaald uit '{HPO_EXPERIMENT_NAME}'")
    for run in runs:
        train_and_log_model(params=dict(run.data.params), data_path=data_path)

    # Selecteer het model met de laagste test RMSE
    best_models_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=best_models_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"Beste model: run_id={run_id}, RMSE={best_run.data.metrics['test_rmse']:.2f}")

    mlflow.register_model(model_uri, name="wind-forecaster-best-model")


if __name__ == "__main__":
    print("...registering model")
    run_register_model("../data/", 5)
