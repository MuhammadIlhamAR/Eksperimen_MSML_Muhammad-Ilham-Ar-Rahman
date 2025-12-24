import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

def train_tuned_model(input_file, n_estimators, max_depth, min_samples_split, mlflow_tracking_uri, experiment_name):
    # Set up MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 1. Load Data
    df = pd.read_csv(input_file)
    
    # 2. Persiapan fitur dan target (Sesuai preprocessing lo)
    X = df.drop(['RAIN_Category'], axis=1)
    y = df['RAIN_Category'].map({'Tidak Hujan': 0, 'Hujan': 1})

    # 3. Split Data (Stratified agar balance) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. MLflow Tracking Run
    with mlflow.start_run(run_name="Tuned_RandomForest_Production"):
        # Log Hyperparameters hasil tuning 
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": 42
        })

        # Inisialisasi Model dengan Parameter Tuning
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth != 0 else None,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        # Training
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        # Plot & Log Confusion Matrix 
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=['Tidak Hujan', 'Hujan'], yticklabels=['Tidak Hujan', 'Hujan'])
        plt.title('Confusion Matrix - Tuned Model')
        plt.savefig("confusion_matrix_tuned.png")
        mlflow.log_artifact("confusion_matrix_tuned.png")

        # Log Model ke DagsHub
        mlflow.sklearn.log_model(model, "best_random_forest_model")
        print(f"Training Selesai! Accuracy: {acc:.4f} | ROC AUC: {roc_auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="PRSA_Data_Aotizhongxin_preprocessing.csv")
    parser.add_argument('--n_estimators', type=int, default=200) # Contoh hasil tuning
    parser.add_argument('--max_depth', type=int, default=20)      # Contoh hasil tuning
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--mlflow_tracking_uri', type=str)
    parser.add_argument('--experiment_name', type=str, default="Tugas Akhir MSML")
    args = parser.parse_args()

    train_tuned_model(
        args.input_file, args.n_estimators, args.max_depth, 
        args.min_samples_split, args.mlflow_tracking_uri, args.experiment_name
    )
