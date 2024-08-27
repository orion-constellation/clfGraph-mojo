import torch
import torch.nn as nn
import torch.nn.functional as F
from math import simd_matrix_multiply


class ISCGatingNetwork(nn.Module):
    def __init__(self, embedding_dim: int, number_of_experts: int):
        super(ISCGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, number_of_experts)
        self.softmax = nn.softmax(dim=-1)
        
    
    def forward(self, shared_representation: torch.Tensor):
        x = torch.relu(self.fc1(shared_representation))
        gating_probs = torch.empty(x.size())
        simd_matrix_multiply(x.detach().numpy(), self.fc2.weight.detach().numpy(), gating_probs.numpy(), x.size(0))
        
