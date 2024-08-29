'''
Attention Layers and additional

'''
import torch
import torch.nn.functional as F
from torch.nn import Module as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree

'''Global Graph Attention Layer
-  Captures information about the entire graph
'''
class GlobalAttentionGNN(nn.Module):
    def __init__(self, node_features, global_features):
        super().__init__()
        self.gnn = GNNLayer(global_features, node_features)
        self.global_attention = nn.Linear(node_features, 1)
        
    def forward(self, x, edge_index, batch):
        h = self.gnn(x, edge_index)
        weights = self.global_attention(h).squeeze()
        weights = scatter_softmax(weights, batch, dim=0)
        return scatter_add(weights.unsqueeze(-1) * h, batch, dim=0)

'''
Node Level Attention:
- Captures information about the node's neighbors
- For use after aggregation
'''
class NodeAttentionGNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gnn = GNNLayer(in_features, out_features)
        self.attention = nn.MultiheadAttention(out_features, num_heads=4)
        
    def forward(self, x, edge_index):
        h = self.gnn(x, edge_index)
        h, _ = self.attention(h, h, h)
        return h


'''Edge Level Attention:
- Implement during message passinf

'''
class AttentionGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
    def forward(self, x, edge_index):
        h = self.W(x)
        edge_h = h[edge_index]
        edge_e = self.a(torch.cat([edge_h[0], edge_h[1]], dim=-1)).squeeze()
        attention = F.softmax(edge_e, dim=0)
        return scatter_add(attention * edge_h[1], edge_index[0], dim=0, dim_size=x.size(0))

