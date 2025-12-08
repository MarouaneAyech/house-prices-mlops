import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
import mlflow
import mlflow.sklearn
import dagshub # La magie opère ici
import yaml

def train_model(data_dir="data/processed", output_dir="models"):

    # 1. Charger params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # 2. Initialiser la connexion DAGsHub (Configuration Automatique)
    # Cela configure le MLFLOW_TRACKING_URI automatiquement
    dagshub.init(repo_owner=params["mlflow"]["repo_owner"], 
                 repo_name=params["mlflow"]["repo_name"], 
                 mlflow=True)
    
    # Configurer l'expérience
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    print("Début de l'entraînement avec MLflow...")
    
    # 3. Démarrer le Run MLflow
    with mlflow.start_run() as run:
        # --- A. Chargement Données ---
        os.makedirs(output_dir, exist_ok=True)
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv")

        # --- B. Entraîner le modèle
        model = LinearRegression()
        model.fit(X_train, y_train)

        # --- C. Logging des Hyperparamètres ---
        # On loggue tout ce qui vient de 'process' et 'train'
        mlflow.log_params(params["data_process"])
        mlflow.log_params(params["train"])

        # --- D. Sauvegarde Locale (Pour DVC) ---
        with open(f"{output_dir}/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # --- E. VERSIONING DU MODÈLE (Model Registry) ---
        # C'est cette ligne qui crée la version dans l'onglet "Models" de DAGsHub
        # mlflow.sklearn.log_model(
        #     sk_model=model, 
        #     artifact_path="model", 
        #     # registered_model_name=params["mlflow"]["registered_model_name"]
        # )
        mlflow.log_artifact(f"{output_dir}/model.pkl", artifact_path="model")
        
        print(f"Modèle entraîné et versionné dans MLflow (Run ID: {run.info.run_id})")

if __name__ == "__main__":
    train_model()
