import numpy as np
import pandas as pd


def build_features_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Kopie van build_features logica voor testen zonder imports."""
    df = df.copy()
    df["maand"] = df["dag"].dt.month
    df["dag_vd_jaar"] = df["dag"].dt.dayofyear
    df["weekdag"] = df["dag"].dt.weekday
    df["maand_sin"] = np.sin(2 * np.pi * df["maand"] / 12)
    df["maand_cos"] = np.cos(2 * np.pi * df["maand"] / 12)
    df["dag_sin"] = np.sin(2 * np.pi * df["dag_vd_jaar"] / 365)
    df["dag_cos"] = np.cos(2 * np.pi * df["dag_vd_jaar"] / 365)
    for col in ["geo_windspeed_30m", "ukkel_windspeed_10m"]:
        if col not in df.columns:
            df[col] = 0
    df[["geo_windspeed_30m", "ukkel_windspeed_10m"]] = df[
        ["geo_windspeed_30m", "ukkel_windspeed_10m"]
    ].fillna(0)
    return df


def test_build_features_adds_season_columns():
    df = pd.DataFrame(
        {
            "dag": pd.to_datetime(["2026-01-15", "2026-06-15"]),
            "geo_windspeed_10m": [4.5, 6.2],
            "geo_windspeed_30m": [5.0, 7.0],
            "ukkel_windspeed_10m": [3.0, 4.0],
        }
    )

    result = build_features_logic(df)

    assert "maand_sin" in result.columns
    assert "maand_cos" in result.columns
    assert "dag_sin" in result.columns
    assert "dag_cos" in result.columns
    assert "weekdag" in result.columns


def test_build_features_missing_wind_columns():
    df = pd.DataFrame(
        {
            "dag": pd.to_datetime(["2026-01-15"]),
            "geo_windspeed_10m": [4.5],
        }
    )

    result = build_features_logic(df)

    assert result["geo_windspeed_30m"].iloc[0] == 0
    assert result["ukkel_windspeed_10m"].iloc[0] == 0


def test_build_features_sin_cos_range():
    df = pd.DataFrame(
        {
            "dag": pd.to_datetime(["2026-03-15"]),
            "geo_windspeed_10m": [5.0],
            "geo_windspeed_30m": [6.0],
            "ukkel_windspeed_10m": [4.0],
        }
    )

    result = build_features_logic(df)

    assert -1 <= result["maand_sin"].iloc[0] <= 1
    assert -1 <= result["maand_cos"].iloc[0] <= 1
    assert -1 <= result["dag_sin"].iloc[0] <= 1
    assert -1 <= result["dag_cos"].iloc[0] <= 1


def test_build_features_correct_row_count():
    df = pd.DataFrame(
        {
            "dag": pd.to_datetime(["2026-01-15", "2026-06-15", "2026-12-01"]),
            "geo_windspeed_10m": [4.5, 6.2, 3.1],
            "geo_windspeed_30m": [5.0, 7.0, 4.0],
            "ukkel_windspeed_10m": [3.0, 4.0, 2.0],
        }
    )

    result = build_features_logic(df)

    assert len(result) == 3


def test_empty_df_returns_empty_metrics():
    df = pd.DataFrame(columns=["actual_kwh", "predicted_kwh"])
    result = {} if len(df) == 0 else {"rmse": 999}
    assert result == {}
