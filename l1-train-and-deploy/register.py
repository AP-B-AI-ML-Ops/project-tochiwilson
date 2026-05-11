import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from prefect import flow, task

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]

mlflow.set_tracking_uri("http://localhost:5000")


@task
def train_and_log_model(params, data_path):
    # Laad en splits data exact zoals in hpo.py
    df = pd.read_csv(os.path.join(data_path, "../data/train_data_wind.csv"))
    X = df[["geo_windspeed_10m"]]
    y = df["elia wind kwh"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        # Train het model met de beste gevonden parameters
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("test_rmse", rmse)

        mlflow.sklearn.log_model(rf, artifact_path="model")


@flow
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

    for run in runs:
        train_and_log_model(params=run.data.params, data_path=data_path)

    # Selecteer het model met de laagste test RMSE uit de nieuwe experimenten
    best_models_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=best_models_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    # Registreer het allerbeste model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="wind-forecaster-best-model")


if __name__ == "__main__":
    print("...registering model")
    run_register_model("../data/", 5)
