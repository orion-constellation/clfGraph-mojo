''' Training Loop Including WandB Code
- Weights and Biases base logging will continue to expand



'''

import os
from dataclasses import dataclass

import torch
import wandb
from dotenv import load_dotenv
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import create_session_id


@dataclass
class TrainingSession:
    session_id: str
    model: nn.Module
    optimizer: optim.Optimizer
    dataloader: DataLoader
    data: torch.Tensor
    num_epochs: int
    device: str = 'cpu'
    project_name: str = "default-project"  # W&B project name
    entity_name: str = None  # W&B entity name

def new_session():
    if session_id == False:
        if os.getenv("WANDB_API_KEY"):
            load_dotenv()
            
        session_id = create_session_id()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        return session_id
    else:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        session_id = session_id
        return session_id




class TrainingSession:
    def __init__(self, session_id, model, optimizer, dataloader, data, num_epochs, device, project_name, entity_name):
        self.session_id = session_id
        self.model = model


def train_model(session: TrainingSession):
    # Initialize W&B
    wandb.init(
        project=session.project_name,
        entity=session.entity_name,
        config={
            "epochs": session.num_epochs,
            "batch_size": session.dataloader.batch_size,
            "learning_rate": session.optimizer.param_groups[0]['lr'],
            "session_id": session.session_id
        }
    )
    
    session.model.to(session.device)
    
    for epoch in range(session.num_epochs):
        session.model.train()
        
        for batch_idx, (inputs, labels) in enumerate(session.dataloader):
            inputs, labels = inputs.to(session.device), labels.to(session.device)
            
            session.optimizer.zero_grad()
            outputs = session.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            session.optimizer.step()
            
            # Log the loss to W&B
            wandb.log({"loss": loss.item(), "epoch": epoch})
            
            print(f"Epoch [{epoch+1}/{session.num_epochs}], Batch [{batch_idx+1}/{len(session.dataloader)}], Loss: {loss.item():.4f}")
    
    print("Training complete!")

    # Save the model checkpoint as an artifact
    torch.save(session.model.state_dict(), f"{session.session_id}_model.pth")
    wandb.save(f"{session.session_id}_model.pth")
    
    # Optionally, log the model as an artifact
    model_artifact = wandb.Artifact(f"{session.session_id}_model", type="model")
    model_artifact.add_file(f"{session.session_id}_model.pth")
    wandb.log_artifact(model_artifact)
    
    wandb.finish()


def train_model_sweep():
    wandb.init(project=PROJECT, name="sweep_session_{wand.run}")
    wandb.sweep(sweep_config)
    
    sweep_config = {
    'method': 'bayes',  # Choose 'grid', 'random', or 'bayes' for Bayesian optimization
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="example-project")

# Example usage:
if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))  # Example model
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataloader = DataLoader(torch.rand((100, 10)), batch_size=32)  # Example data
    data = torch.rand((100, 10))  # Example data
    
    session = TrainingSession(
        session_id="001",
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        data=data,
        num_epochs=10,
        device='cpu',
        project_name="example-project",
        entity_name="example-entity"
    )
    
    train_model(session)