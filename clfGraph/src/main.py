'''
MoE Model:
- Integrate Custom Operations and Set Classification and Graph Layers
-Create the training loop passing data between each model



'''
import torch
from torch.nn import Module as nn
from torch_geometric.nn import GATConv, GCNConv 
from clf_design import MoEGate, 


# @TODO To be integrated
# Instantiate the experts
experts = [ExpertGNN(num_node_features=16, num_classes=3) for _ in range(num_experts)]



class MoEModel(nn.Module):
    def __init__(self, gate, experts, custom_ops_path):
        super(MoEModel, self).__init__()
        self.gate = gate
        self.experts = experts
        self.custom_ops_path = custom_ops_path

    def forward(self, x, edge_index, context):
        # Use the gate to select the expert
        expert_weights = self.gate(x)
        selected_expert = torch.argmax(expert_weights, dim=-1)

        # Extract relevant subgraph using the custom op
        subgraph = torch.ops.custom.subgraph_extract(x, context)

        # Pass the subgraph to the selected expert
        output = self.experts[selected_expert](subgraph, edge_index)
        return output

# Instantiate the MoE model
moe_model = MoEModel(gate=gate, experts=experts, custom_ops_path="custom_ops.mojopkg")

# Training Loop
optimizer = torch.optim.Adam(moe_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    moe_model.train()
    for data in dataloader:
        x, edge_index, context = data.x, data.edge_index, data.context
        optimizer.zero_grad()
        output = moe_model(x, edge_index, context)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()

# Benchmark the model with Max
max benchmark onnx_model.onnx --custom-ops-path=custom_ops.mojopkg