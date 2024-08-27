'''
# Shared Representation Layer 
### Based on STIX2.1 and the MITRE ATT&CK Framework
- Implemented using PyTorch and PyTorch Geometric




'''
import torch
import torch.nn as nn 

class SharedRepresentation(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SharedRepresentation, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        shared_representation = self.fc2(x)
        return shared_representation