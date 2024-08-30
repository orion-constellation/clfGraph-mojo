from math import simd_matrix_multiply

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

'''
Shared Representation:
- Generates a representation of the STIX Data Chosen
- Will begin with a simple model and tune with the data



'''

class SharedRepresentation(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SharedRepresentation, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        shared_representation = self.fc2(x)
        return shared_representation
    


    
class ISCGatingNetwork(nn.Module):
    def __init__(self, embedding_dim: int, number_of_experts: int):
        super(ISCGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, number_of_experts)
        self.softmax = F.softmax(self(embedding_dim, dim=-1))
        
    
    def forward(self, shared_representation: torch.Tensor):
        x = torch.relu(self.fc1(shared_representation))
        gating_probs = torch.empty(x.size())
        simd_matrix_multiply(x.detach().numpy(), self.fc2.weight.detach().numpy(), gating_probs.numpy(), x.size(0))
        

class GNNLayer:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.weights = [1.0] * (self.n_features * self.n_features)
        self.bias = [0.0] * self.n_features
        
    def forward(self, n_features: int, adjancey_matrix: torch.Tensor):
        output = [0.0] * (self.n_features * self.n_features)
        simd_matrix_multiply(self.n_features, self.weights, output, self.n_features)
        
        # Add bias and apply activation function
        output = [x + 1.0 for x in output]  # Example activation
        
        return output


'''
Hierarchical MoE Base Class:
- To be revised and tuned when the GNN is implemented
    - GNN will be the MoE

'''
    
    
    
class HMoE:
    def __init__(self, classifier_weights, n_experts: int, n_features: int):
        self.classifier_weights = classifier_weights
        self.gnn_experts = [GNNLayer(n_features) for _ in range(n_experts)]
        
    def forward(self, x)

class MoEGate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super(MoEGate, self).__init__()
        self.gc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)





