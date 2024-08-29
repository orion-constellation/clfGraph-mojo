# ClfGraph
An experimental project exploring Graph Neural Networks and Mixture of Experts architectures for use on threat data.
Baseline results come from an array of Sci-kit Learn models and their results.

Potential for further explorations by arranging the dataset into a time-series dataset and streaming it.

Trained on the CICIoMT2024 "Attacks" Dataset available here:
http://205.174.165.80/IOTDataset/CICIoMT2024/Dataset/WiFI_and_MQTT/attacks

# Research Goal

Incorporate a Classification Gate into a Hiearchal Mixture of Experts

# Hiearachal Mixture of Experts (HMoE) Model

Research and Development of Composite Architecture enhanced by custom SIMD
and Operations for expanding knowledge of structuring MITRE Attack as Graphs
as well as subsampling using GNNs.

This is purely a fun exploration and learning activity both in the space of GNN's

**It can also serve as a WandB logging sklearn template on it's own by extracting the sklearn_baseline**



1. Classifier Gated
2. Master Graph is composed of MITRE Attack Graph
3. Sub-sampling using GNN (Creates Experts)

## Use of Mojo
1. Custom Attention Op
2. Matrix Multiplication SIMD
3. Improving systems programming knowledge