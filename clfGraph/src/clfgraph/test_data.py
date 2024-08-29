import pandas as pd
from collections import defaultdict

from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

def test_data():
    data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Score': [85.5, 90.3, 88.0, 72.5, 91.2]
    }

    df = pd.DataFrame(data)
    return df

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