import numpy as np
import pandas as pd


def prepare_data(
    wind_path: str = "../data/wind_final.csv",
    prod_path: str = "../data/productie_combined.csv",
    output_path: str = "../data/train_data_wind.csv",
) -> pd.DataFrame:
    """
    Verwerkt ruwe wind- en productiedata naar een ML-ready dataset.

    Strategie:
    - Wind is DAGELIJKS -> productie aggregeren naar daggemiddelde
    - Seizoensfeatures toevoegen (maand, dag van het jaar, sin/cos cyclisch)
    - Meerdere windkolommen behouden als features waar beschikbaar

    Returns: opgeschoonde DataFrame klaar voor training
    """

    # 1. Laden
    df_wind = pd.read_csv(wind_path)
    df_prod = pd.read_csv(prod_path)

    # 2. Tijdskolommen normaliseren
    df_wind["dag"] = pd.to_datetime(df_wind["date"]).dt.normalize()
    df_prod["tijd"] = pd.to_datetime(df_prod["tijd"]).dt.tz_localize(None)
    df_prod["dag"] = df_prod["tijd"].dt.normalize()

    # 3. Productie aggregeren naar dagniveau
    #    - gemiddelde per dag (representatief voor typisch uurprofiel)
    #    - som per dag (totale dagproductie)
    prod_dag = (
        df_prod.groupby("dag")
        .agg(
            elia_wind_kwh_gemiddeld=("elia wind kwh", "mean"),
            elia_wind_kwh_totaal=("elia wind kwh", "sum"),
            elia_zon_kwh_gemiddeld=("elia zon kwh", "mean"),
        )
        .reset_index()
    )

    # 4. Wind op dagniveau bewaren (al dagelijks)
    wind_features = ["dag", "geo_windspeed_10m", "geo_windspeed_30m", "ukkel_windspeed_10m"]
    df_wind_dag = df_wind[wind_features].copy()

    # 5. Merge op dag
    df_merged = pd.merge(df_wind_dag, prod_dag, on="dag", how="inner")

    # 6. Seizoensfeatures toevoegen
    #    - Cyclische encoding (sin/cos) zodat dec en jan dicht bij elkaar liggen
    df_merged["maand"] = df_merged["dag"].dt.month
    df_merged["dag_vd_jaar"] = df_merged["dag"].dt.dayofyear
    df_merged["weekdag"] = df_merged["dag"].dt.weekday

    # Cyclische encoding voor maand en dag van het jaar
    df_merged["maand_sin"] = np.sin(2 * np.pi * df_merged["maand"] / 12)
    df_merged["maand_cos"] = np.cos(2 * np.pi * df_merged["maand"] / 12)
    df_merged["dag_sin"] = np.sin(2 * np.pi * df_merged["dag_vd_jaar"] / 365)
    df_merged["dag_cos"] = np.cos(2 * np.pi * df_merged["dag_vd_jaar"] / 365)

    # 7. Opschonen: verwijder rijen waar de hoofdfeature of target ontbreekt
    verplichte_kolommen = ["geo_windspeed_10m", "elia_wind_kwh_gemiddeld"]
    df_clean = df_merged.dropna(subset=verplichte_kolommen).copy()

    # 8. Debug info
    print("--- DEBUG INFO ---")
    print(f"Rijen na merge:      {len(df_merged)}")
    print(f"Rijen na dropna:     {len(df_clean)}")
    print("\nNon-null counts per kolom:")
    print(df_clean.count())
    print("\nBeschrijvende statistieken target:")
    print(df_clean["elia_wind_kwh_gemiddeld"].describe())
    print("------------------\n")

    # 9. Opslaan
    df_clean.to_csv(output_path, index=False)
    print(f"Opgeschoonde dataset: {len(df_clean)} rijen opgeslagen naar {output_path}")

    return df_clean


if __name__ == "__main__":
    df = prepare_data()
    print("\nEerste rijen:")
    print(df.head())
    print("\nFeature kolommen beschikbaar voor training:")
    feature_cols = [
        "geo_windspeed_10m",
        "geo_windspeed_30m",
        "ukkel_windspeed_10m",
        "maand_sin",
        "maand_cos",
        "dag_sin",
        "dag_cos",
        "weekdag",
    ]
    print(feature_cols)
