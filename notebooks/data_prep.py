import pandas as pd


def prepare_data():
    # 1. Laad de bestanden in
    df_wind = pd.read_csv("../data/wind_final.csv")
    df_prod = pd.read_csv("../data/productie_combined.csv")

    # 2. Tijdskolommen gelijktrekken
    df_wind["tijd"] = pd.to_datetime(df_wind["date"])
    df_prod["tijd"] = pd.to_datetime(df_prod["tijd"]).dt.tz_localize(None)

    # 3. Merge
    df_merged = pd.merge(df_wind, df_prod, on="tijd", how="inner")

    # DEBUG: Laten we kijken wat we nu hebben
    print(f"--- DEBUG INFO ---")
    print(f"Aantal rijen na de samenvoeging (merge): {len(df_merged)}")
    print("Hoeveel niet-lege (geldige) waardes heeft elke kolom?")
    print(
        df_merged[
            ["tijd", "geo_windspeed_10m", "ecmwf_windspeed_10m", "elia wind kwh"]
        ].count()
    )
    print(f"------------------\n")

    # 4. Kies de kolommen met de beste data
    # We proberen nu geo_windspeed_10m omdat ecmwf waarschijnlijk leeg is
    features_and_target = ["tijd", "geo_windspeed_10m", "elia wind kwh"]

    df_clean = df_merged[features_and_target].dropna()

    df_clean.to_csv("../data/train_data_wind.csv", index=False)
    print(f"Eindresultaat opgeschoonde dataset: {len(df_clean)} rijen klaar voor ML!")

    return df_clean


if __name__ == "__main__":
    df = prepare_data()
    print(df.head())
