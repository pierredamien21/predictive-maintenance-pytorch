"""Neural network models for predictive maintenance."""

import torch
from torch import nn


class MLP(nn.Module):
    """Baseline multilayer perceptron using the last timestep."""

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs[:, -1, :]).squeeze(-1)


class RecurrentRegressor(nn.Module):
    """LSTM or GRU regressor for remaining useful life."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell: str = "lstm",
    ) -> None:
        super().__init__()
        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.recurrent = recurrent(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        return self.head(outputs[:, -1, :]).squeeze(-1)


class MultiTaskModel(RecurrentRegressor):
    """Recurrent model with RUL regression and failure classification heads."""

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__(input_size, hidden_size)
        self.failure_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.recurrent(inputs)
        features = outputs[:, -1, :]
        return self.head(features).squeeze(-1), self.failure_head(features).squeeze(-1)
