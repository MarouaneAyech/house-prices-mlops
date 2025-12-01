import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import mlflow

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):

    # MLFLOW 1 : Chargement du Run ID (Pont)
    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()

    # MLFLOW 2 : Reprise du Run MLflow
    with mlflow.start_run(run_id=run_id):

        X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        y_test = pd.read_csv(f"{data_dir}/y_test.csv")
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv")

        # with open(model_path, "rb") as f:
        #     model = pickle.load(f)
        #MLFLOW 3 : Chargement du Modèle depuis l'Artifact Store du Run (Récupération)
        model_uri = f"runs:/{run_id}/model" 
        model = mlflow.sklearn.load_model(model_uri)

        y_train_pred = model.predict(X_train)
        mse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_train = r2_score(y_train, y_train_pred)
        print(f"Train MSE: {mse_train:.2f}")
        print(f"Train R²: {r2_train:.3f}")

        y_test_pred = model.predict(X_test)
        mse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_test = r2_score(y_test, y_test_pred)
        print(f"Test MSE: {mse_test:.2f}")
        print(f"Test R²: {r2_test:.3f}")

        # MLFLOW 4 : LOG DES MÉTRIQUES MINIMALES
        mlflow.log_metric("MSE_Train", mse_train)
        mlflow.log_metric("R2_Train", r2_train)
        mlflow.log_metric("MSE_Test", mse_test)
        mlflow.log_metric("R2_Test", r2_test)

        print(f"Métriques loguées au Run ID: {run_id}")

if __name__ == "__main__":
    evaluate_model()
