import os
import pandas as pd
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from prefect import task

mlflow.set_tracking_uri("http://localhost:5000")


@task
def run_optimization(data_path: str, num_trials: int):
    mlflow.set_experiment("random-forest-hyperopt")
    mlflow.sklearn.autolog(disable=True)

    # 1. Laad de data en splits deze
    df = pd.read_csv(os.path.join(data_path, "../data/train_data_wind.csv"))
    X = df[["geo_windspeed_10m"]]
    y = df["elia wind kwh"].values

    # We gebruiken een vaste random_state voor reproduceerbaarheid
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)

            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    print("...optimizing params")
    run_optimization("../data/", 5)
