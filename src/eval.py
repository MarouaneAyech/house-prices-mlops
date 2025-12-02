import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import json

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl", output_dir='metrics'):
    os.makedirs(output_dir, exist_ok=True)
    
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    print(f"Train RMSE: {rmse_train:.2f}")
    print(f"Train R²: {r2_train:.3f}")

    y_test_pred = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    print(f"Test RMSE: {rmse_test:.2f}")
    print(f"Test R²: {r2_test:.3f}")

    
    # with open(f"{output_dir}/metrics.txt","w") as file :
    #     file.write(f"Train RMSE: {rmse_train:.2f}\n")
    #     file.write(f"Train R²: {r2_train:.3f}\n")
    #     file.write(f"Test RMSE: {rmse_test:.2f}\n")
    #     file.write(f"Test R²: {r2_test:.3f}\n")

    metrics = {
        "train_rmse": rmse_train,
        "train_r2": r2_train,
        "test_rmse": rmse_test,
        "test_r2": r2_test
    }

    # Sauvegarde en JSON
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_model()
