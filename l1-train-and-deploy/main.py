from prefect import flow

# Importeer de flows/functies uit je andere bestanden
from preprocess import prepare_data
from train import run_train
from hpo import run_optimization
from register import run_register_model


@flow(name="Wind Energy Training Pipeline")
def training_flow():
    # 1. Maak de data schoon (dit genereert train_data_wind.csv)
    prepare_data()

    # 2. Train de simpele baseline (Linear Regression)
    run_train()

    # 3. Zoek naar de beste Random Forest parameters
    # Zoek 5 combinaties, en geef aan waar de data staat
    run_optimization("../data/", 5)

    # 4. Registreer het beste model
    run_register_model("../data/", 5)


if __name__ == "__main__":
    training_flow()
