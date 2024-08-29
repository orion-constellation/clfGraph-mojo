'''
Simple  Streamlit Graph to Vizualise Data




'''

import os
from collections import defaultdict
from datetime import date

import pandas as pd
import streamlit as st
import wandb
from constants import PROJECT_NAME
from sklearn.base import BaseEstimator
from sklearn_baseline.main import clf_models, cluster_models, params
from sklearn_baseline.models import (train_classification_models,
                                     train_clustering_models)


# Function to ensure results directory exists
def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to load CSV files from the data directory
def load_data_sources():
    data_dir = "../data/dataset/final"
    
    if not os.path.exists(data_dir):
        st.error(f"No data folder found. Please add some CSV files to the data directory.")
        return None
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        st.error("No CSV files found in the data directory.")
        return None
    
    return {os.path.splitext(f)[0]: os.path.join(data_dir, f) for f in csv_files}

# Initialize Streamlit
def main():
    st.title("ClfGraph Dashboard")
    st.subheader("by Nentropy")

    with st.sidebar:
        st.header("Configuration")
        
        # Project selection dropdown
        project_selection = st.selectbox("Select Project", ["clfGraph"])
        
        # Load available data sources
        data_sources = load_data_sources()
        if data_sources:
            selected_source = st.selectbox(
                "Select a Data Source:",
                options=list(data_sources.keys()),
            )

            # Load the selected data source
            data_path = data_sources[selected_source]
            #data = pd.read_csv(data_path)
            st.session_state['data'] = data

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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Classification"):
                # Placeholder for training data split
                models = clf_models
                params = params
                train_classification_models(data=data, models=models, params=params)
        
        with col2:
            if st.button("Run Clustering"):
                # Placeholder for clustering data split
                models = cluster_models
                params = params
                train_clustering_models(data, cluster_models, params)
        
        # Display results if available
        if st.session_state['classification_results'] or st.session_state['clustering_results']:
            st.write("### Results")
            if st.session_state['classification_results']:
                st.write(st.session_state['classification_results'])
            if st.session_state['clustering_results']:
                st.write(st.session_state['clustering_results'])
        
        # Button to view results in Weights and Biases
        if st.button("View Results in W&B"):
            wandb.init(project="clfGraph")
            st.write("Opening Weights and Biases...")
            wandb.run.log_code(".")
            wandb.finish()

if __name__ == "__main__":
    st.set_page_config(
        page_title="ClfGraph Dashboard", page_icon=":chart_with_upwards_trend:"
    )
    main()
