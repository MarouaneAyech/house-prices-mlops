import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_path)
    features=df.describe(include=['number']).columns
    df = df[features].dropna()

    # Keep Only numeric features most correlated with the target
    # X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    features_sorted=df.corr().iloc[:,-1].abs().sort_values()[::-1][1:]
    X=df.loc[:,features_sorted[features_sorted>0.5].index]
    
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y_binned)
    
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Données préparées et sauvegardées !")

if __name__ == "__main__":
    prepare_data()
