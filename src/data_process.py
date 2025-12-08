import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

def prepare_data(input_path="data/raw/data.csv", output_dir="data/processed"):

    # 1. Charger la config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["data_process"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep only not nan numeric data
    df = pd.read_csv(input_path)
    features=df.describe(include=['number']).columns
    df = df[features].dropna()

    # Split int representative train & test data
    y = df["SalePrice"]
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    df_train, df_test = train_test_split(df, 
                                         test_size=params["test_size"], 
                                         random_state=params["random_state"], 
                                         stratify=y_binned)
    # get target
    y_train = df_train['SalePrice']    
    y_test = df_test['SalePrice'] 

    # Keep only most correlated features with target
    features_sorted=df_train.corr().iloc[:,-1].abs().sort_values()[::-1][1:]
    
    correlation_threashold = params["correlation_threshold"] 
    X_train=df_train.loc[:,features_sorted[features_sorted>correlation_threashold].index]
    X_test=df_test.loc[:,features_sorted[features_sorted>correlation_threashold].index]
    

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Données préparées et sauvegardées !")

if __name__ == "__main__":
    
    prepare_data()
