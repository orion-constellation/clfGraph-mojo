import os
from collections import defaultdict

import pandas as pd
import streamlit as st
import wandb
from clfgraph.sklearn_baseline.models import (save_model,
from clfgraph.test_data import clf_models, cluster_models                                            train_classification_models,
                                              train_clustering_models)
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# Import the training functions (replace with actual import statements)



# Function to ensure results directory exists
def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to load CSV files from the data directory
def load_data_sources():
    data_dir = "final"
    
    if not os.path.exists(data_dir):
        st.error(f"No data folder found. Please add some CSV files to the '{data_dir}' directory.")
        return None
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        st.error("No CSV files found in the data directory.")
        return None
    
    data_sources = {os.path.splitext(f)[0]: os.path.join(data_dir, f) for f in csv_files}
    return data_sources

def main():
    st.title("Custom HMoE Model Data Dashboard")
    st.subheader("by Nentropy")

    with st.sidebar:
        st.header("Configuration")
        
        # Project selection dropdown
        project_select = st.selectbox("Select Project", ["clfGraph"])
        if project_select not in st.session_state:
            st.session_state[project_select] = project_select
        
        # Load available data sources
        data_sources = load_data_sources()
        if data_sources:
            selected_source = st.selectbox(
                "Select a Data Source:",
                options=list(data_sources.keys()),
            )

            # Load the selected data source and store it in session_state
            data_path = data_sources[selected_source]
            st.session_state['data'] = pd.read_csv(data_path)
        else:
            st.warning("No data sources available to select.")


        # Model configuration
        if 'data' in st.session_state:
            data = st.session_state['data']
            st.dataframe(data.head())

        # Placeholder for storing model training results
        if 'classification_results' not in st.session_state:
            st.session_state['classification_results'] = None

        if 'clustering_results' not in st.session_state:
            st.session_state['clustering_results'] = None

        # Model training buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Classify"):
                clf_models = clf_models
                params = {
                    "project_name": "clfGraph",
                    "random_state": 42,
                    "test_size": 0.2
                }
                with st.spinner:
                    st.session_state["classification results"] = train_classification_models(data, clf_models, params)
            
        with col2:
            if st.button("Cluster"):
                cluster_models = defaultdict(BaseEstimator, {
                    "KMeans": KMeans(),
                    "DBSCAN": DBSCAN(),
                    "AgglomerativeClustering": AgglomerativeClustering(),
                    "MeanShift": MeanShift()
                })
                params = {
                    "project_name": "clfGraph",
                    "random_state": 42,
                    "n_clusters": 3
                }
                with st.spinner:
                    st.session_state["clustering results"] = train_clustering_models(data, cluster_models, params)
                    st.cache_resource
        
        with col3:
            if st.button("Save Model"):
                
                cluster_models=cluster_models
                params = {
                    "project_name": "clfGraph",
                    "random_state": 42,
                    "n_clusters": 3
                }
                with st.spinner:
                    st.session_state["clustering results"] = train_clustering_models(data, cluster_models, params)
                    
                
                    
        
        # Display results if available
        if st.session_state['classification_results'] or st.session_state['clustering_results']:
            st.write("### Results")
            if st.session_state['classification_results']:
                st.write(st.session_state['classification_results'])
            if st.session_state['clustering_results']:
                st.write(st.session_state['clustering_results'])
                
            col3, col4 = st.columns(2)
            
            with col3, col4:
                if col3:
                    st.button("Refresh")
                    st.session_state.clear()
                if col4:
                    st.button("Toggle Results")
                    if "classification results" | "clustering results" not in st.session_state:
                        st.write("Click classify or cluster")
                        
                        
                
        
        # Button to view results in Weights and Biases
        if st.button("View Results in W&B"):
            st.write("Opening Weights and Biases...")
            st.page_link(f"https://www.wandb.ai/orionai/{PROJECT_NAME}")
        
        st.
            

st.set_page_config(
        page_title="ClfGraph Dashboard", page_icon=":chart_with_upwards_trend:"
    )
main()
    