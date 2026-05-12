import os
import pandas as pd
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from prefect import task

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


@task
def run_optimization(data_path: str, num_trials: int):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("random-forest-hyperopt")
    mlflow.sklearn.autolog(disable=True)

    # 1. Laad de data
    df = pd.read_csv(os.path.join(data_path, "train_data_wind.csv"))

    # Gebruik alleen aanwezige features, vul ontbrekende waarden op met mediaan
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].fillna(df[available_features].median())
    y = df[TARGET].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200, step=10),
            "max_depth": trial.suggest_int("max_depth", 2, 20, step=1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, step=1),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("features", available_features)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)

            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

    print(f"\nBeste trial: RMSE = {study.best_value:.2f} kWh")
    print(f"Beste params: {study.best_params}")


if __name__ == "__main__":
    print("...optimizing params")
    run_optimization("../data/", 20)
