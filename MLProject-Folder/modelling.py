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
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Load dataframe
    df = pd.read_csv(input_file)

    # Determine target column
    if 'RAIN_Category' in df.columns:
        y = df['RAIN_Category']
    elif 'RAIN' in df.columns:
        y = np.where(df['RAIN'] == 0, 'Tidak Hujan', 'Hujan')
    else:
        raise ValueError("Input file must contain either 'RAIN_Category' or 'RAIN' column")

    # Map to numeric
    if y.dtype == object or y.dtype.name == 'category':
        y = pd.Series(y).map({'Tidak Hujan': 0, 'Hujan': 1})

    # Features: drop obvious non-feature columns
    drop_cols = ['RAIN', 'RAIN_Category', 'year', 'hour', 'No', 'station']
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
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
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', acc)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Hujan', 'Hujan'], yticklabels=['Tidak Hujan', 'Hujan'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        mlflow.log_artifact('confusion_matrix.png')

        # Save model
        mlflow.sklearn.log_model(model, 'random_forest_model')
        print(f"Accuracy: {acc:.4f}")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None)
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
