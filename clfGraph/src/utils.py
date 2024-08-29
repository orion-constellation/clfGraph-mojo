'''
# Utils 
### Based on STIX2.1 and the MITRE ATT&CK Framework
- Implemented using PyTorch and PyTorch Geometric
- Creates a shared representation of the input data for use by the network
'''
import os
from uuid import uuid4

import joblib
import streamlit as st
from clfgraph.constants import PROJECT_NAME


def create_session_id():
    return str(uuid4())







