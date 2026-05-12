import pandas as pd
import numpy as np
from datetime import date


def build_features_predict(data: dict) -> pd.DataFrame:
    """Kopie van build_features uit predict.py voor testen zonder MLflow."""
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
    return pd.DataFrame([row])


def test_predict_build_features_returns_dataframe():
    data = {
        "datum": "2026-05-12",
        "geo_windspeed_10m": 4.5,
        "geo_windspeed_30m": 5.2,
        "ukkel_windspeed_10m": 3.8,
    }
    result = build_features_predict(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_predict_build_features_correct_columns():
    data = {"datum": "2026-05-12", "geo_windspeed_10m": 4.5}
    result = build_features_predict(data)

    expected_cols = [
        "geo_windspeed_10m",
        "geo_windspeed_30m",
        "ukkel_windspeed_10m",
        "maand_sin",
        "maand_cos",
        "dag_sin",
        "dag_cos",
        "weekdag",
    ]
    for col in expected_cols:
        assert col in result.columns


def test_predict_build_features_defaults_missing_wind():
    data = {"datum": "2026-05-12", "geo_windspeed_10m": 4.5}
    result = build_features_predict(data)

    assert result["geo_windspeed_30m"].iloc[0] == 0
    assert result["ukkel_windspeed_10m"].iloc[0] == 0
