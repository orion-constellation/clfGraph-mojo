'''
Using SKLearn models to get a baseline for the task of categorizing data as a threat or not
    Purpose: Comparison against PyTorch Model as well as a template for other datasets
    and projects







'''

import os
from collections import defaultdict
from datetime import date
from typing import List

import matplotlib as plt
import pandas as pd
import pyarrow as pa
import seaborn as sns
import wandb
from constants import PROJECT_NAME
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.custom_logging import configure_logging

'''
Train classification models using the specified classifier models on the provided training and testing data.

Parameters:
- clf_models: A dictionary containing the classifier models to train.
- X_train: The training features.
- y_train: The training labels.
- X_test: The testing features.
- y_test: The testing labels.
- params: A dictionary containing additional parameters, including the project name.

Returns:
None
'''
def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

params = {
    "project_name": PROJECT_NAME,
    "random_state": 42,
    "test_size": 0.2,
    "n_clusters": 3,
    "svc_kernel": "rbf",
    "svc_c": 1.0,
    "logistic_regression_c": 1.0,
    "max_depth_decision_tree": None,
    "n_estimators_random_forest": 100,
    "learning_rate_gradient_boosting": 0.1
}

wandb.config = params

def get_data_df(file_path):
    df = pd.read_csv(data)
    df_process = df.copy()
    
    # Create feature data frame assuming preprocessing has been completed, and a target.
    y = df_process.drop('target')
    X = df.drop('target')
    
    return X, y

'''
Train classification data:

Params:
Input Data: CSV
clf_models: defaultdict[str, BaseEstimator]
params: global scope dict

returns Data: Dataframe for X, y

'''

def train_classification_models(data: pd.DataFrame, clf_models: defaultdict[str, BaseEstimator], params: dict):
    
    project=PROJECT_NAME
    df_train = pd.read_csv(f"data/final/{PROJECT_NAME}_train.csv")
    df_test = pd.read_csv(f"data/final/{PROJECT_NAME}_test.csv")

    
    df_input = df_train.copy()
    X = df_input.drop(columns=["target"])
    y = df_train["target"]
    X, y = get_data_df(file_path="../data/dataset/final/CICIoMT2024_final_dataset_0x0")
    
    # Split Dataset
    X_train, y_train, X_test, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
    
    project_name = params.get("project_name", "classification_project")
    today_str = pa.timestamp('ms').now().to_pandas().strftime("%Y-%m-%d")
    results_dir = f"./results/{today_str}/"
    ensure_dir(results_dir)

    for model_name, model in clf_models.items():
        # Initialize W&B with the parameters
        wandb.init(project=project_name, name=f"{model_name}_{today_str}", config=params)
        
        # Update model parameters from the params dictionary if necessary
        if model_name == "SVC":
            model.set_params(kernel=params["svc_kernel"], C=params["svc_c"])
        elif model_name == "LogisticRegression":
            model.set_params(C=params["logistic_regression_c"])
        elif model_name == "DecisionTree":
            model.set_params(max_depth=params["max_depth_decision_tree"])
        elif model_name == "RandomForest":
            model.set_params(n_estimators=params["n_estimators_random_forest"])
        elif model_name == "GradientBoosting":
            model.set_params(learning_rate=params["learning_rate_gradient_boosting"])

        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
        f1_score = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        try:
            wandb.Artifact(f"{model_name}_confusion", type="confusion_matrix")
        except:
            logger.error(f"Error saving artifact for: {model_name}")
        
        # Log metrics to W&B
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "f1_score": f1_score,
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=y_pred, class_names=["0", "1"])
        })

        # Save results to CSV
        results_df = pd.DataFrame({
            "Model": [model_name],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "ROC AUC": [roc_auc]
        })
        results_file = os.path.join(results_dir, f"{project_name}_{model_name}_{today_str}_results.csv")
        huggingface_push
        results_df.to_csv(results_file, index=False)

        # Save the model artifact
        wandb.save(results_file)
        wandb.finish()


'''Train clustering models using the specified cluster models on the provided training data.

Parameters:
- cluster_models: A dictionary containing the cluster models to train.
- X_train: The training data.
- params: A dictionary containing additional parameters, including the project name.

Returns:
None'''

def train_clustering_models(data, cluster_models: defaultdict[str, BaseEstimator], params: dict):
    project= PROJECT_NAME
    df_train = pd.read_csv(f"data/final/{PROJECT_NAME}_train.csv")
    df_test = pd.read_csv(f"data/final/{PROJECT_NAME}_test.csv")

    
    df_input = df_train.copy()
    X = df_input.drop(columns=["target"])
    y = df_train["target"]
    
    if os.path.exists("../data/dataset/final/CICIoMT2024_final_dataset_0x0.csv"):
        X, y = get_data_df(file_path="../data/dataset/final/CICIoMT2024_final_dataset_0x0.csv")
    
    # Split Dataset
    X_train, y_train, X_test, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
    
    project_name = params.get("project_name", "clustering_project")
    today_str = pa.timestamp('ms').now().to_pandas().strftime("%Y-%m-%d")
    results_dir = f"./results/{today_str}/"
    ensure_dir(results_dir)

    for model_name, model in cluster_models.items():
        # Initialize W&B with the parameters
        wandb.init(project=project_name, name=f"{model_name}_{today_str}", config=params)
        
        # Update model parameters from the params dictionary if necessary
        if model_name == "KMeans":
            model.set_params(n_clusters=params["n_clusters"])

        # Train the model
        model.fit(X_train)
        
        # Predictions and metrics
        cluster_labels = model.predict(X_train)
        inertia = model.inertia_ if hasattr(model, "inertia_") else None
        silhouette_avg = silhouette_score(X_train, cluster_labels) if hasattr(model, "inertia_") else None
        
        # Log metrics to W&B
        wandb.log({
            "inertia": inertia,
            "silhouette_score": silhouette_avg,
            "cluster_centers": wandb.Table(data=model.cluster_centers_, columns=X_train.columns) if hasattr(model, "cluster_centers_") else None
        })

        # Save results to CSV
        results_df = pd.DataFrame({
            "Model": [model_name],
            "Inertia": [inertia],
            "Silhouette Score": [silhouette_avg]
        })
        results_file = os.path.join(results_dir, f"{project_name}_{model_name}_{today_str}_results.csv")
        results_df.to_csv(results_file, index=False)

        # Save the model artifact
        wandb.save(results_file)
        wandb.finish()
        
