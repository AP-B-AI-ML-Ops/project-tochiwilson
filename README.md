# Renewable Energy Forecasting MLOps Project

## Project Explanation

### What does your service actually do?

This project is an end-to-end Machine Learning system that predicts renewable energy production (solar and wind in MW) for the Antwerp region over the next 24 hours. It uses short-term weather forecasts as input to generate accurate predictions.

### What kind of application are you making?

We are building a **dual-deployment MLOps system** consisting of:

* **Web Service**: A containerized REST API (FastAPI) that accepts on-demand weather forecast data and instantly returns predicted energy production.
* **Batch Service**: An automated, scheduled pipeline (Prefect) that periodically fetches new weather forecasts, runs inference, stores predictions, and monitors model performance over time.

### What is the goal of this project?

The transition to renewable energy introduces grid instability because solar and wind production are highly dependent on weather conditions and difficult to predict.

The goal of this project is to provide reliable short-term forecasts that:

* Help grid operators make balancing decisions and prevent blackouts
* Enable prosumers to optimize energy usage
* Support energy traders in making informed decisions on the Day-Ahead market

---

## Flows & Actions

To adhere to MLOps best practices, we implement automated flows using **Prefect** and **MLflow**.

---

### Flow 1: Model Training Pipeline

**Action 1 (Extract)**
Load historical weather and energy production data from a PostgreSQL database.

**Action 2 (Transform)**

* Join datasets on the `tijd` (datetime) column
* Handle missing values
* Extract time-based features (e.g., hour of day, month)

**Action 3 (Train & Track)**

* Train a Machine Learning model
* Use MLflow to track experiments
* Log parameters and evaluation metrics

**Action 4 (Register)**
Register the best-performing model in the MLflow Model Registry.

---

### Flow 2: On-Demand Inference (Web Service)

**Action 1**
Load the production-ready model from the MLflow registry into a FastAPI application.

**Action 2**
Expose a `/predict` endpoint that accepts a JSON payload containing weather forecast parameters.

**Action 3**
Return predicted energy production (MW) for the requested timeframe.

---

### Flow 3: Batch Inference & Monitoring Pipeline (Scheduled)

**Action 1 (Fetch Forecast)**
Call the Open-Meteo ECMWF API to retrieve weather forecasts for the next 24 hours.

**Action 2 (Batch Predict)**
Run inference using the registered model and store predictions in the database.

**Action 3 (Fetch Actuals)**
Retrieve actual energy production data from the Elia API.

**Action 4 (Evaluate & Monitor)**

* Compare predictions with actual values
* Calculate error metrics (e.g., RMSE)
* Send results to Evidently and Grafana for monitoring dashboards

**Action 5 (Trigger Retraining)**
If performance drops below a predefined threshold (data drift or model decay), automatically trigger the retraining pipeline (Flow 1).

---

## Summary

This project demonstrates a full MLOps lifecycle:

* Data ingestion and preprocessing
* Model training and experiment tracking
* Model deployment (real-time + batch)
* Continuous monitoring and automated retraining

It provides a scalable and production-ready solution for renewable energy forecasting.
