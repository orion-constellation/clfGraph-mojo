# ClfGraph

Trained on the CICIoMT2024 "Attacks" Dataset available here:
http://205.174.165.80/IOTDataset/CICIoMT2024/Dataset/WiFI_and_MQTT/attacks


![Version](https://img.shields.io/badge/version-v0.1.0-blue)
![Release](https://img.shields.io/badge/release-latest-green)
![Tests](https://github.com/orion-constellation/clfGraph-mojo/actions/workflows/ci.yml/badge.svg?branch=main&event=push)

## Overview

### Research Goal

Incorporate a Classification Gate into a Hiearchal Mixture of Experts

### Hiearachal Mixture of Experts (HMoE) Model

Research and Development of Composite Architecture enhanced by custom SIMD
and Operations for expanding knowledge of structuring MITRE Attack as Graphs
as well as subsampling using GNNs.

This is purely a fun exploration and learning activity both in the space of GNN's

**It can also serve as a WandB logging sklearn template on it's own by extracting the sklearn_baseline**


ClfGraph is a research project focused on developing a model architecture that leverages MITRE data to create a shared representation layer, coupled with a classifier to gate a Hierarchical Mixture of Experts (HMoE) GCN and GAT models. The aim is to periodically build fingerprints of the information using graphs. The project integrates various components and uses Weights and Biases for experiment tracking.

## Features

- 🧠 **Streamlit Front-End**: Interactive web interface for visualizing and managing models.
- 📊 **Baseline & Advanced Analysis**: Scikit-learn baseline models alongside deep learning with PyTorch.
- 🗂️ **Database Integration**: Supports both PostgreSQL and Memgraph for data handling.
- 🔥 **Rust & Python Interoperability**: Rust for high-performance Memgraph interactions, Python for flexible data processing and modeling.
- 🚀 **Dockerized Environment**: Multi-service architecture for streamlined development and deployment.

## Use of Mojo & Rust 🦀🔥

1. Custom Attention Op
2. Matrix Multiplication SIMD
3. Memgraph Client and operations in Rust
4. Improving systems programming knowledge

## How to Contribute

We welcome contributions to ClfGraph! Whether you're interested in refining the model architecture, enhancing the front-end, or improving database interactions, your input is valuable.

### Steps to Contribute:

1. Fork the repository.
2. Clone your fork locally.
3. Create a new branch for your feature or bugfix.
4. Submit a pull request with a clear description of your changes.

Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details.

Happy coding! 😊