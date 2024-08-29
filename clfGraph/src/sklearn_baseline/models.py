'''
Using SKLearn models to get a baseline for the task of categorizing data as a threat or not
    Purpose: Comparison against PyTorch Model as well as a template for other datasets
    and projects

'''

import os
from collections import defaultdict
from datetime import date
from typing import List, Union

import dask
import dask.dataframe as dd
import huggingface_hub
import matplotlib as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
import seaborn as sns
import streamlit as st
import torch

from clfgraph.test_data import test_data_df
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login
from sklearn.base import BaseEstimator

import wandb

load_dotenv("sklearn.env")
from clfgraph.test_data import test_data_df, clf_models, cluster_models
from clfgraph.constants import __VERSION__, DATA_PATH, PROJECT_NAME, WANDB_MODE
from clfgraph.custom_logging import configure_logging
from clfgraph.hf_hub import upload_to_hf_hub
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             pairwise_distances_argmin_min, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from dask import dataframe, datasets

task=("clf", "clustering")
logger = configure_logging(level = "INFO")
device = "mps" if torch.backends.mps.is_available() else "cpu"
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

def save_model(project_name=PROJECT_NAME, model_type=Union["clf", "cluster"], task=Union["clf", "clusster"] ):
    # Define the directory path
    save_dir = f"./results/models/{project_name}/sklearn/"
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    model_name=model_name
    # Define the file path
    


params = {
    "project_name": PROJECT_NAME,
    "model_name": "model_x",
    "clf_task": "clf",
    "cluster_task": "cluster",
    "random_state": 42,
    "test_size": 0.2,
    "n_clusters": 3,
    "svc_kernel": "rbf",
    "svc_c": 1.0,
    "learning_rate": 0.01,
    "logistic_regression_c": 1.0,
    "max_depth_decision_tree": 4,
    "n_estimators_random_forest": 100,
    "learning_rate_gradient_boosting": 0.1
}
#Init WandB
def init_wandb(config, name, project=PROJECT_NAME, job_type="model"):
        return wandb.init(config=params, name="clf model training")

#@FIXME Do I need this?
def get_data_df():
    #df = pd.read_csv(file_path)
    
    data, df, X, y = test_data_df()
    df_process = df.copy()
    logger.info("read files in from: {%s}", df_process)
    
    # Create feature data frame assuming preprocessing has been completed, and a target.
    
    return (X, y)


X, y = get_data_df()

'''
Train classification data:

Params:
Input Data: CSV
clf_models: defaultdict[str, BaseEstimator]
params: global scope dict

returns Data: Dataframe for X, y

'''
@st.cache_resource
def train_classification_models(clf_models: defaultdict[str, BaseEstimator], params=params, task = "clf"):
    task="clf"
    init_wandb(project=PROJECT_NAME, config=params, name="classifier set of training", job_type="model")
    model_name=f"final_v{__VERSION__}_{model_type}_{task}_0x0"
    if logger.level != "DEBUG":
        #df_train = pd.read_csv(f"../data/final/{PROJECT_NAME}_train.csv")
        #df_test = pd.read_csv(f"../data/final/{PROJECT_NAME}_test.csv")
        #X, y = get_data_df(file_path="../data/dataset/final/CICIoMT2024_final_dataset_0x0.csv")
        X, y = test_data_df(data)

    
    
    #Log to WandB
    wandb.log({"features": X, "target": y})
    feature_table = wandb.Table(dataframe=X, rows=15)
    wandb.log({"feature_table":feature_table})
    wandb.log_artifact(feature_table)
    

    # Split Dataset
    X_train, y_train, X_test, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    
    project_name = params.get("project_name", "classification_project")
    today_str = pa.timestamp('ms').now().to_pandas().strftime("%Y-%m-%d")
    results_dir = f"./results/{today_str}/"
    ensure_dir(results_dir)

    for model_type, model in tqdm(clf_models.items()):
        # Initialize W&B with the parameters
        run_name = f"{model_type}_run_{today_str}"
        run_name = wandb.init(project=project_name, name=run_name, config=params)
        model_name = model.set_params(model_name=params["name"])
        
        # Update model parameters from the params dictionary if necessary
        if model_type == "SVC":
            model.set_params(kernel=params["svc_kernel"], C=params["svc_c"])
        elif model_type == "LogisticRegression":
            model.set_params(C=params["logistic_regression_c"])
        elif model_type == "DecisionTree":
            model.set_params(max_depth=params["max_depth_decision_tree"])
        elif model_type == "RandomForest":
            model.set_params(n_estimators=params["n_estimators_random_forest"])
        elif model_type == "GradientBoosting":
            model.set_params(learning_rate=params["learning_rate_gradient_boosting"])

        # Train the model
        model = model.fit(X_train, y_train)
        q_model = input("Do you want to save the {model_type} model? Y/n")
        
        if q_model == "Y" or "y":
            try:        
                save_model(model)
                logger.info(f'{model_name} saved')
            except RuntimeError as e:
                logger.error("Error:\n {}", e)
            
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
        f1_score = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_file = conf_matrix.to_csv(f"./results/{project_name}_clf_{pa.date32(today_str)}_0x0.csv")
        try:
            clf_conf = wandb.Artifact(conf_matrix, name=f"{model_name}_confusion")
            clf_conf.add_file(conf_file)
            wandb.log_artifact(clf_conf)
            
        except Exception as e:
            logger.error(f"Error saving artifact for: {model_name}: {e}")
        
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
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "f1_score": f1_score,
            "ROC AUC": roc_auc
        })
        
        results_file = os.path.join(results_dir, f"{project_name}_{model_name}_{today_str}_results.csv")
        login(token=os.getenv("HF_HUB_TOKEN"), add_to_git_credential=True)
        
        
        
        results_df.to_csv(results_file, index=False)
        upload_to_hf_hub(params=params, name=model_name, __VERSION__=__VERSION__)
        # Save the model artifact
        wandb.save(results_file)
        wandb.finish()
        return results_df, model
        test_data = test_data()
'''Distributed Training'''
def train_dask_ray(data=test_data_df, partitions=60, clf_models=clf_models):
    # Initialize Ray
    ray.init()

    # Start a Dask client with Ray as the scheduler
    client = Client(scheduler="ray")

    # Load a dataset
    data, df, X, y = test_data_df()

    # Convert to Dask DataFrame
    ddf = dd.from_array(X)

    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a scikit-learn model in parallel using Ray
    @ray.remote
    def train_model(data, X_train, y_train, model_type="clf",clf_models=clf_models, partitions=60):
        for _, model in enumerate(clf_models):
            model.fit(X_train, y_train)
        

        # Train the model
        for model in tqdm(clf_models=clf_models):
            model_ref = train_model.remote(X_train, y_train)
            model = ray.get(model_ref) 
            save_q = input("Do you want to save the mode?")
            if save_q == "Yes" or "y" or "yes":
                save_model(PROJECT_NAME, model_type==model, task="clf")
            return model, model_ref
#@TODO Finish the remote tranining

'''Train clustering models using the specified cluster models on the provided training data.

Parameters:
- cluster_models: A dictionary containing the cluster models to train.
- X_train: The training data.
- params: A dictionary containing additional parameters, including the project name.

Returns:
None'''

@st.cache_resource
def train_clustering_models(data, _cluster_models: defaultdict[str, BaseEstimator], params=params):
    task="clustering"
    project= PROJECT_NAME
    model_name=f"test_v{__VERSION__}_{task}_0x0"
    if logger.level != "DEBUG":
        model_name=f"final_v{__VERSION__}_{task}_0x0"
        #df_train = pd.read_csv(f"data/final/{PROJECT_NAME}_train.csv")
        #df_test = pd.read_csv(f"data/final/{PROJECT_NAME}_test.csv")

    wandb.init(project=PROJECT_NAME, name="cluster model training", job_type="model", config=params)
    wandb.config = params
    logger.setLevel("INFO")
    dataset_path="data/dataset/train/"
    if logger.level != "DEBUG":
        #df_train = pd.read_csv(f"{dataset_path}/final/{PROJECT_NAME}_train.csv")
        #df_test = pd.read_csv(f"../data/final/{PROJECT_NAME}_test.csv")
        #X, y = get_data_df(file_path=f"{dataset_path}final/CICIoMT2024_final_dataset_0x0.csv")
        X, y = test_data() #get_data_df()

    
    
    if os.path.exists("../data/dataset/final/CICIoMT2024_final_dataset_0x0.csv"):
        X, y = get_data_df(file_path="../data/dataset/final/CICIoMT2024_final_dataset_0x0.csv")
    
    # Split Dataset
    X_train, y_train, X_test, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
    
    project_name = params.get("project_name", "clustering_project")
    today_str = pa.timestamp('ms').now().to_pandas().strftime("%Y-%m-%d")
    results_dir = f"./results/{today_str}/"
    ensure_dir(results_dir)

    for model_type, model in cluster_models.items():
        # Initialize W&B with the parameters
        wandb.init(project=project_name, name=f"{model_name}_{today_str}", config=params)
        
        # Update model parameters from the params dictionary if necessary
        if model_type == "KMeans":
            model.set_params(n_clusters=params["n_clusters"])
            kmeans = KMeans(n_clusters=4, random_state=42)
            cluster_labels = kmeans.fit_predict(X_train)
            

            # Modify centroids based on labeled data
            for i, label in tqdm(enumerate(np.unique(y))):
                X_labeled = X[y == label]
                cluster_centers = kmeans.cluster_centers_
                nearest, _ = pairwise_distances_argmin_min(cluster_centers, X_labeled)
                kmeans.cluster_centers_[i] = np.mean(X_labeled[nearest], axis=0)
            
            pred = model.fit_predict(X_test, y_test)
            # Re-fit with adjusted centers
            

        # Train the model
            cluster_model = model.fit(X_train)
            cluster_artifact = wandb.Artifact(f"{model_type}_v{__VERSION__}_0x0")
                                              
            
            y_pred = model.predict(X_test, y_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        #precision = precision_score(y_test, y_pred)
        #recall = recall_score(y_test, y_pred)
        #roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
        #f1_score = f1_score(y_test, y_pred)
        # Predictions and metrics
        cluster_labels = model.predict(X_train)
        accuracy = accuracy_score(y_test, y_pred)
        inertia = model.inertia_ if hasattr(model, "inertia_") else None
        silhouette_avg = silhouette_score(X_train, cluster_labels, random_state=42, sample_size=500) if hasattr(model, "inertia_") else None
        
        # Log metrics to W&B
        wandb.log({
            "accuracy": accuracy,
            "inertia": inertia,
            "silhouette_score": silhouette_avg,
            "cluster_centers": wandb.Table(data=model.cluster_centers_, columns=X_train.columns) if hasattr(model, "cluster_centers_") else None
        })

        # Save results to CSV
        results_df = pd.DataFrame({
            "Model": model_name,
            "Inertia": inertia,
            "Silhouette Score": silhouette_avg,
            "inertia": inertia,
            "accuracy": accuracy,
        })
        results_file = os.path.join(results_dir, f"{project_name}_{model_name}_{today_str}_results.csv")
        results_df.to_csv(results_file, index=False)

        # Save the model artifact
        wandb.save(results_file)
        wandb.finish()
       
try:
    if os.getenv("TASK") == "clf":
        logger.info("Starting clf training")
        tqdm(train_classification_models(clf_models=clf_models, data=test_data, params=params, task="clf"),desc="clf models training... ", mininterval=0.5, ascii=True)
    elif os.getenv("TASK")=="cluster":
        logger.info("Starting cluster training...")
        tqdm(train_clustering_models(data=test_data_df,_cluster_models=cluster_models, params=params),desc="starting clustering...", ascii=True, mininterval=0.5, colour="red")
    else: 
        logger.info("Starting clf & cluster cluster training...")
        logger.info("Clister first...")
        tqdm(train_clustering_models(data=test_data_df,_cluster_models=cluster_models, params=params),desc="starting clustering...", ascii=True, mininterval=0.5, colour="red")
        logger.info("Moving onto clf...")
except Exception as e:
    logger.error(e, exc_info=True)