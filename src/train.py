import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
import mlflow

def train_model(data_dir="data/processed", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    # MLFLOW 1 : Démarrage du Run MLflow
    with mlflow.start_run(run_name="Linear_Regression") as run:
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv")

        model = LinearRegression()
        model.fit(X_train, y_train)

        # MLFLOW 2 : Log des Hyperparamètres
        mlflow.log_param("model_type", "LinearRegression")
        
        # Garder model.pkl pour pouvoir le charger dans eval.py !
        # with open(f"{output_dir}/model.pkl", "wb") as f:
        #     pickle.dump(model, f) 
        # print("Modèle entraîné et sauvegardé.")

        # MLFLOW 3 : Sauvegarder le Run ID pour eval.py
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        # MLFLOW 4 : Enregistrement du Modèle dans MLflow Registry (Optionnel mais recommandé)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="SalesPrice_Predictor"
        )
        print(f"Run MLflow démarré: {run_id}")
        

if __name__ == "__main__":
    train_model()
