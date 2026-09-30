import torch
import torch.nn as nn
from omegaconf import DictConfig


class RegressionModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)