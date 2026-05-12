# Wind Energy Production Forecasting

An end-to-end MLOps system that predicts wind energy production (kWh) for the Antwerp region using weather forecast data as input.

## Problem Description

The renewable energy transition creates grid instability: solar and wind production are weather-dependent and inherently difficult to predict. Grid operators, energy traders, and prosumers all need reliable short-term forecasts to make balancing decisions.

This project builds an ML system that predicts the **average daily wind energy production (kWh)** for the Antwerp region, using historical wind speed measurements from multiple weather stations.

### Inputs

| Feature | Description |
|---|---|
| `geo_windspeed_10m` | Wind speed at 10m height (km/h) — Geo.be station |
| `geo_windspeed_30m` | Wind speed at 30m height (km/h) — Geo.be station |
| `ukkel_windspeed_10m` | Wind speed at 10m height (km/h) — Ukkel station |
| `maand_sin` / `maand_cos` | Cyclical encoding of the month |
| `dag_sin` / `dag_cos` | Cyclical encoding of the day of year |
| `weekdag` | Day of the week (0=Monday, 6=Sunday) |

### Output

| Output | Description |
|---|---|
| `predicted_kwh` | Predicted average wind energy production (kWh/day) |
| `predicted_mwh` | Same value converted to MWh |

### Why is this useful?

Accurate wind energy forecasts allow grid operators to balance supply and demand, energy traders to optimize buying and selling decisions, and prosumers to plan their energy consumption. This system provides 24-hour ahead forecasts using live ECMWF weather data.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   docker-compose                     │
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ database │  │  experiment  │  │ orchestration │  │
│  │ postgres │  │  tracking    │  │   prefect     │  │
│  └──────────┘  │   mlflow     │  └───────────────┘  │
│                └──────────────┘                      │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │   l1-train-deploy   │  │    l2-web-service      │ │
│  │   training pipeline │  │    Flask REST API      │ │
│  └─────────────────────┘  └────────────────────────┘ │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │  l4-batch-service   │  │       grafana          │ │
│  │  batch predictions  │  │    monitoring UI       │ │
│  │  + monitoring       │  └────────────────────────┘ │
│  └─────────────────────┘                             │
└─────────────────────────────────────────────────────┘
```

---

## Technologies

| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking & model registry |
| Prefect | Workflow orchestration |
| Evidently | Model performance monitoring |
| Grafana | Metrics visualization |
| PostgreSQL | Backend for MLflow & metrics storage |
| Docker | Containerization |
| Flask | REST API for web service |

---

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- The following data files in the `data/` folder:
  - `wind_final.csv`
  - `productie_combined.csv`

---

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd project-tochiwilson
```

### 2. Start the infrastructure

```bash
docker compose up database experiment-tracking orchestration grafana
```

Wait until you see:
- MLflow: `Listening at: http://0.0.0.0:5000`
- Prefect: `Check out the dashboard at http://0.0.0.0:4200`

### 3. Train the model

```bash
docker compose --profile training run --rm train-deploy
```

This runs the full training pipeline:
- Data preprocessing
- Linear Regression baseline
- Random Forest hyperparameter optimization (Optuna)
- Model registration in MLflow

### 4. Start the web service

```bash
docker compose up web-service
```

The API is available at `http://localhost:9696`.

**Example request:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "datum": "2026-05-12",
    "geo_windspeed_10m": 4.5,
    "geo_windspeed_30m": 5.2,
    "ukkel_windspeed_10m": 3.8
  }'
```

**Example response:**
```json
{
  "predicted_kwh": 223428.43,
  "predicted_mwh": 223.428
}
```

### 5. Run the batch service

```bash
docker compose --profile batch run --rm batch-service
```

This will:
- Fetch recent weather forecasts
- Run inference with the registered model
- Compare predictions against Elia actuals
- Generate an Evidently HTML report in `data/batch_results/`
- Store metrics in PostgreSQL
- Log metrics to MLflow
- Trigger retraining if RMSE exceeds the threshold (150,000 kWh)

### 6. Monitor in Grafana

Open `http://localhost:3400` (login: admin/admin).

Connect to the PostgreSQL data source:
- Host: `database:5432`
- Database: `metrics`
- User: `admin`
- Password: `admin`

Query to visualize RMSE over time:
```sql
SELECT run_time AS time, value::float AS rmse
FROM evidently_metrics
WHERE metric_name = 'rmse'
ORDER BY run_time
```

---

## Project Structure

```
project-tochiwilson/
├── backend-database/           # PostgreSQL setup
│   ├── Dockerfile
│   ├── .env
│   └── init.sql
├── l1-train-and-deploy/        # Training pipeline
│   ├── main.py                 # Prefect flow orchestration
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Linear Regression baseline
│   ├── hpo.py                  # Hyperparameter optimization
│   ├── register.py             # Model registration
│   ├── Dockerfile
│   └── requirements.txt
├── l2-web-service/             # REST API
│   ├── predict.py              # Flask API
│   ├── Dockerfile
│   └── requirements.txt
├── l3-backend-orchestration/   # Prefect server
│   └── Dockerfile
├── l4-batch-service/           # Batch inference & monitoring
│   ├── batch.py                # Prefect batch flow
│   ├── retrain.py              # Retraining trigger
│   ├── Dockerfile
│   └── requirements.txt
├── tests/                      # Unit tests
│   ├── test_batch.py
│   ├── test_predict.py
│   └── test_preprocess.py
├── data/                       # Data files (not in git)
├── docker-compose.yml
├── pyproject.toml
├── .pre-commit-config.yaml
└── .flake8
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Model Performance

| Model | RMSE |
|---|---|
| Linear Regression (baseline) | ~58,620 kWh |
| Random Forest (optimized) | ~37,914 kWh |

The Random Forest model uses Optuna for hyperparameter optimization and is trained on daily wind speed data from 2025-2026.

---

## Dependencies

All dependencies are pinned in the respective `requirements.txt` files per service:

| Service | Key dependencies |
|---|---|
| train-deploy | mlflow==2.16.0, scikit-learn==1.7.1, prefect==3.4.0, optuna==3.6.1 |
| web-service | flask==3.0.3, mlflow==2.16.0, scikit-learn==1.7.1 |
| batch-service | mlflow==2.16.0, scikit-learn==1.4.2, evidently==0.4.30, prefect==3.4.0 |
