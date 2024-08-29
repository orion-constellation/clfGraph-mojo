import os

import wandb
from constants import PROJECT_NAME, WANDB_MODE
from models import train_classification_models, train_clustering_models
from sklearn.base import BaseEstimator

os.environ(WANDB_MODE) = "disabled"



clf_models = defaultdict(BaseEstimator, {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVC": SVC(probability=True)
})

cluster_models = defaultdict(BaseEstimator, {
    "KMeans": KMeans(),
    "DBSCAN": DBSCAN(),
    "AgglomerativeClustering": AgglomerativeClustering(),
    "MeanShift": MeanShift()
})

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

def train_models(clf_models=clf_models, cluster_models=clustermodels, params= params, type="classification"):
    try:
        if type == classification:
            train_classification_models(clf_models, params)
        elif
            train_clustering_models(cluster_models, params)