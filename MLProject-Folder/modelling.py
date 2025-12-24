import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

def train_model(input_file, n_estimators=100, random_state=42, test_size=0.2, mlflow_tracking_uri=None, experiment_name="experiment"):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    mlflow.set_experiment(experiment_name)

    # Load data
    df = pd.read_csv(input_file)
    
    # Target logic
    if 'RAIN_Category' in df.columns:
        y = df['RAIN_Category']
    else:
        y = np.where(df['RAIN'] == 0, 'Tidak Hujan', 'Hujan')
    
    y = pd.Series(y).map({'Tidak Hujan': 0, 'Hujan': 1})

    # Feature logic
    drop_cols = ['RAIN', 'RAIN_Category', 'year', 'hour', 'No', 'station']
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1).select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    with mlflow.start_run(run_name="RandomForest_Training"):
        mlflow.log_param('n_estimators', n_estimators)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric('accuracy', acc)
        
        # Plotting
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        mlflow.sklearn.log_model(model, 'random_forest_model')
        print(f"Success! Accuracy: {acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--mlflow_tracking_uri', type=str)
    parser.add_argument('--experiment_name', type=str, default='Tugas Akhir MSML')
    args = parser.parse_args()

    train_model(args.input_file, args.n_estimators, args.random_state, args.test_size, args.mlflow_tracking_uri, args.experiment_name)
