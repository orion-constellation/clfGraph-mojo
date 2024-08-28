'''
Graph Neural Network Components

'''
import torch
from torch_geometric.nn import GATConv, GCNConv
from torch.nn import Module as nn
from torch.nn import functional as F

class GNNLayer(nn.Module):
    def __init__(self, input_dim: int, number_of_experts:int):
        super(GNNLayer, self).__init__()
        self.gate = MoEGate(input_dim, number_of_experts)
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(number_of_experts)])

    def forward(self, x):
        gate_outputs = self.gate(x)
        expert_outputs = [expert(x) for expert in self.experts]
        return torch.sum(torch.stack(expert_outputs) * gate_outputs, dim=0)
    

'''
GNN Expert Design:
- Use after the shared representation is generated once as the first layeer

'''
class GNNExpert(nn.Module):
    def __init__(self, embedding_dim:int, heading_dim: int, num_classes: int, heads=8):
        super(GNNExpert, self).__init__()
        self.conv1 = GATConv(embedding_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads)
        self.fc = nn.Linear(hidden_dim * heads, num_classes)
        
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        
        attention_result = torch.empty(x.size())
        simd_attention(x.detach().numpy(), self.fc2.weight.detach().numpy(), attention_result.numpy()) 
        attention_result = torch.tensor(attention_result)

        x = F.relu(self.conv2(attention_result, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return self.log_softmax(x, dim=1)

