import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

def train_model(input_file, n_estimators=100, random_state=42, test_size=0.2, mlflow_tracking_uri=None, experiment_name="experiment"):
    # Konfigurasi MLflow tracking uri jika disediakan
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri) [cite: 1]

    mlflow.set_experiment(experiment_name) [cite: 1]

    # Load dataframe
    df = pd.read_csv(input_file) [cite: 1]

    # Menentukan kolom target
    if 'RAIN_Category' in df.columns:
        y = df['RAIN_Category'] [cite: 1]
    elif 'RAIN' in df.columns:
        # fallback: buat kategori jika hanya ada nilai numerik
        y = np.where(df['RAIN'] == 0, 'Tidak Hujan', 'Hujan') [cite: 1]
    else:
        raise ValueError("Input file must contain either 'RAIN_Category' or 'RAIN' column") [cite: 1]

    # Map target ke numerik (0 dan 1)
    if y.dtype == object or y.dtype.name == 'category':
        y = pd.Series(y).map({'Tidak Hujan': 0, 'Hujan': 1}) [cite: 1]

    # Features: hapus kolom non-fitur
    drop_cols = ['RAIN', 'RAIN_Category', 'year', 'hour', 'No', 'station'] [cite: 1]
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1) [cite: 1]

    # Pastikan hanya tipe numerik yang digunakan
    X = X.select_dtypes(include=[np.number]) [cite: 1]

    # Train/test split dengan stratify agar distribusi kelas seimbang
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    ) [cite: 1]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) [cite: 1]

    with mlflow.start_run(run_name="RandomForest_Training"): [cite: 1]
        # Log parameters
        mlflow.log_param('n_estimators', n_estimators) [cite: 1]
        mlflow.log_param('random_state', random_state) [cite: 1]
        mlflow.log_param('test_size', test_size) [cite: 1]

        # Training
        model.fit(X_train, y_train) [cite: 1]

        # Prediksi
        y_pred = model.predict(X_test) [cite: 1]

        # Log metrics
        acc = accuracy_score(y_test, y_pred) [cite: 1]
        mlflow.log_metric('accuracy', acc) [cite: 1]
        
        # Logging metrik tambahan untuk kelas 0 dan 1
        mlflow.log_metric('precision_class_1', precision_score(y_test, y_pred, pos_label=1)) [cite: 1]
        mlflow.log_metric('recall_class_1', recall_score(y_test, y_pred, pos_label=1)) [cite: 1]

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred) [cite: 1]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Tidak Hujan', 'Hujan'], 
                    yticklabels=['Tidak Hujan', 'Hujan']) [cite: 1]
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(cm_path) [cite: 1]
        plt.close()
        
        # Log artifacts ke MLflow
        mlflow.log_artifact(cm_path) [cite: 1]

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, 'random_forest_model') [cite: 1]

        print(f"Training selesai. Accuracy: {acc:.4f}") [cite: 1]

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RandomForest model for rainfall classification') [cite: 1]
    parser.add_argument('--input_file', type=str, required=True, help='Path to preprocessed CSV file') [cite: 1]
    parser.add_argument('--n_estimators', type=int, default=100) [cite: 1]
    parser.add_argument('--random_state', type=int, default=42) [cite: 1]
    parser.add_argument('--test_size', type=float, default=0.2) [cite: 1]
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None) [cite: 1]
    parser.add_argument('--experiment_name', type=str, default='Tugas Akhir MSML') [cite: 1]
    args = parser.parse_args() [cite: 1]

    train_model(
        input_file=args.input_file,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        test_size=args.test_size,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name
    ) [cite: 1]
