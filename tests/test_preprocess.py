import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../l1-train-and-deploy"))


def test_read_dataframe_returns_dataframe():
    from train import read_dataframe

    df = pd.DataFrame(
        {
            "dag": ["2026-01-15", "2026-01-16"],
            "geo_windspeed_10m": [4.5, 5.0],
            "elia_wind_kwh_gemiddeld": [100000.0, 120000.0],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    result = read_dataframe(tmp_path)

    assert isinstance(result, pd.DataFrame)
    os.unlink(tmp_path)


def test_read_dataframe_parses_dag_column():
    from train import read_dataframe

    df = pd.DataFrame(
        {
            "dag": ["2026-01-15", "2026-01-16"],
            "geo_windspeed_10m": [4.5, 5.0],
            "elia_wind_kwh_gemiddeld": [100000.0, 120000.0],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    result = read_dataframe(tmp_path)

    assert pd.api.types.is_datetime64_any_dtype(result["dag"])
    os.unlink(tmp_path)
