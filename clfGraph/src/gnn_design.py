'''
Graph Neural Network Components

'''
import torch
from torch_geometric.nn import GATConv, GCNConv
from torch.nn import Module as nn
from torch.nn import functional as F
# Import from Mojo
from mojo_math import simd_matrix_multiply, simd_attention



class GNNLayer:
    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = [1.0] * (n_features * n_features)
        self.bias = [0.0] * n_features

    def forward(self, node_features, adjacency_matrix):
        output = [0.0] * (self.n_features * self.n_features)
        simd_matrix_multiply(node_features, self.weights, output, self.n_features)
        
        # Add bias and apply activation function
        output = [x + 1.0 for x in output]  # Example activation
        
        return output

''' 
ExpertGNN:
- Each is a trained on a different subset of data eg. Attacks types

'''

class ExpertGNN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ExpertGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        # Replace with SIMD Attention?
        self.attn = GATConv(128, 128, heads=8)
        self.fc = nn.Linear(128 * 8, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.attn(x, edge_index))
        x = torch.mean(x, dim=0)
        return F.softmax(self.fc(x), dim=-1)



