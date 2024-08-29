'''
MoE Model (Pytorch and Torch Geometric)
- Integrate Custom Operations and Set Classification and Graph Layers
- Create the training loop passing data between each model
- Compare Results to the SKLearn and basic statistical analysis

'''
import os

import torch

from torch
from dataclasses import dataclass

from clf_design import ExpertGNN, MoEGate
from torch.nn import Module as nn
from torch.optim.optimizer import SGD, Adam, Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from utils import create_session_id

import wandb


@dataclass
class TrainingSession:
    session_id: str
    model: nn.Module
    optimizer: Adam, SGD
    dataloader: DataLoader
    data: torch.Tensor
    num_epochs: int
    device: str = "mps" if torch.backends.is_available() else "cpu"
    project_name: str = "default-project"  # W&B project name
    entity_name: str = None  # W&B entity name
    scheduler: lr_scheduler
    learning_rate: float
    loss_fn: nn.Module

def new_session():
    if session_id == False:
        device = try device==device except: "conda: 0"
        session_id = create_session_id()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        return session_id
    else:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        session_id = session_id
        return session_id




# Instantiate the MoE model
moe_model = MoEModel(gate=gate, experts=experts, custom_ops_path=["./mojo_math.mojopkg", "./subgraph.mojopkg", "./data_processing.mojopkg"])
optimizer = torch.optim.Adam(moe_model.parameters(), lr=0.001)
gate = MoEGate()
experts = [ExpertGNN(num_node_features=16, num_classes=3) for _ in range(num_experts)]

def train_model(moe_model, optimizer, dataloader, num_epochs):
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