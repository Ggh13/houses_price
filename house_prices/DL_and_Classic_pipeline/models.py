import torch
import torch.nn as nn
from omegaconf import DictConfig


class RegressionModel(nn.Module):
    """
    MLP-регрессор для предсказания цены дома.
    
    Архитектура: переменное количество скрытых слоёв с BatchNorm, GELU активацией и Dropout.
    Ширина каждого следующего слоя уменьшается вдвое (hidden_dim // 2^i).
    Вход: тензор признаков, выход: скалярное предсказание цены.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        """
        Инициализирует MLP с переменным количеством скрытых слоёв.
        
        Args:
            in_features: Количество входных признаков.
            hidden_dim: Ширина первого скрытого слоя.
            num_layers: Количество слоёв (включая выходной).
            dropout: Вероятность dropout.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        layers = []
        current_dim = in_features

        for i in range(num_layers - 1):
            next_dim = hidden_dim // (2 ** i)
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход. Принимает батч признаков, возвращает предсказания.
        
        Args:
            x: Тензор признаков формы [batch_size, in_features].
        
        Returns:
            Тензор предсказаний формы [batch_size, 1].
        """
        return self.net(x)
