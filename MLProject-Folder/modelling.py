"""
modelling.py
Standalone training script for the PRSA rainfall classification task.
- Reads a preprocessed CSV (expects a 'RAIN_Category' column or numeric 'RAIN')
- Trains a RandomForestClassifier
- Logs parameters, metrics and artifacts to MLflow
- Saves the trained model via mlflow.sklearn.log_model and locally with joblib

Usage examples:
python modelling.py --input_file PRSA_Data_Aotizhongxin_preprocessing.csv --n_estimators 100 --random_state 42 --test_size 0.2 --mlflow_tracking_uri """https://dagshub.com/m.ilham2408/my-first-repo.mlflow""" --experiment_name "Tugas Akhir MSML"

"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib


def train_model(input_file, n_estimators=100, random_state=42, test_size=0.2, mlflow_tracking_uri=None, experiment_name="experiment"):
    # Configure MLflow tracking uri if provided
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Load dataframe
    df = pd.read_csv(input_file)

    # Determine target column
    if 'RAIN_Category' in df.columns:
        y = df['RAIN_Category']
    elif 'RAIN' in df.columns:
        # fallback: create category
        y = np.where(df['RAIN'] == 0, 'Tidak Hujan', 'Hujan')
    else:
        raise ValueError("Input file must contain either 'RAIN_Category' or 'RAIN' column")

    # Map to numeric
    if y.dtype == object or y.dtype.name == 'category':
        y = pd.Series(y).map({'Tidak Hujan': 0, 'Hujan': 1})

    # Features: drop obvious non-feature columns if present
    drop_cols = ['RAIN', 'RAIN_Category', 'year', 'hour', 'No', 'station']
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    # Ensure column types are numeric (if wd_encoded exists keep it)
    # If any non-numeric columns remain, try to one-hot encode or drop them
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        # Simple fallback: drop non-numeric columns
        X = X.drop(columns=non_numeric)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    with mlflow.start_run(run_name="RandomForest_Training"):
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('random_state', random_state)
        mlflow.log_param('test_size', test_size)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_pred_proba = None

        acc = accuracy_score(y_test, y_pred)
        prec0 = precision_score(y_test, y_pred, pos_label=0)
        rec0 = recall_score(y_test, y_pred, pos_label=0)
        f10 = f1_score(y_test, y_pred, pos_label=0)
        prec1 = precision_score(y_test, y_pred, pos_label=1)
        rec1 = recall_score(y_test, y_pred, pos_label=1)
        f11 = f1_score(y_test, y_pred, pos_label=1)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba if y_pred_proba is not None else y_pred)
        except Exception:
            roc_auc = float('nan')

        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision_class_0', prec0)
        mlflow.log_metric('recall_class_0', rec0)
        mlflow.log_metric('f1_class_0', f10)
        mlflow.log_metric('precision_class_1', prec1)
        mlflow.log_metric('recall_class_1', rec1)
        mlflow.log_metric('f1_class_1', f11)
        if not np.isnan(roc_auc):
            mlflow.log_metric('roc_auc', roc_auc)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Hujan', 'Hujan'], yticklabels=['Tidak Hujan', 'Hujan'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # ROC curve if probabilities available
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            roc_path = 'roc_curve.png'
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path)

        # Log the model with mlflow
        mlflow.sklearn.log_model(model, 'random_forest_model')

        # Also save locally
        local_model_path = 'random_forest_model.joblib'
        joblib.dump(model, local_model_path)
        mlflow.log_artifact(local_model_path)

        print(f"Accuracy: {acc:.4f} | ROC AUC: {roc_auc}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RandomForest model for rainfall classification')
    parser.add_argument('--input_file', type=str, required=True, help='Path to preprocessed CSV file')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None, help='Optional MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, default='Tugas Akhir MSML')
    args = parser.parse_args()

    train_model(
        input_file=args.input_file,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        test_size=args.test_size,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name
    )
