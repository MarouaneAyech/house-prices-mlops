import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
import mlflow
import mlflow.sklearn

# import yaml

def train_model(data_dir="data/processed", output_dir="models"):

    # # 1. Charger params
    # with open("params.yaml", "r") as f:
    #     params = yaml.safe_load(f)
        
    # --- A. Chargement Données ---
    os.makedirs(output_dir, exist_ok=True)
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")

    # --- B. Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- C. Sauvegarde Locale (Pour DVC) ---
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Modèle entraîné et sauvegardé !")

if __name__ == "__main__":
    train_model()
