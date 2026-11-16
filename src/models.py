"""Modèles : MLP, LSTM / GRU, et modèle multi-entrées / multi-sorties.

Tous les modèles ont la même signature `forward(seq, ctx)` pour partager la boucle
d'entraînement. Les régresseurs ignorent `ctx` et renvoient la RUL normalisée `(batch,)` ;
`MultiTaskRUL` renvoie `(rul, logits)`.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """Aplatit la fenêtre : ignore l'ordre temporel des cycles."""

    def __init__(
        self, n_features: int, window: int, hidden_size: int = 64, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(seq).squeeze(-1)


class SequenceEncoder(nn.Module):
    """LSTM ou GRU qui résume une fenêtre `(batch, W, features)` en un vecteur."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        cell: str = "lstm",
    ) -> None:
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # n'agit qu'entre les couches
        )
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, state = self.rnn(seq)
        h_n = state[0] if isinstance(state, tuple) else state  # le LSTM renvoie (h_n, c_n)
        # h_n : (couches * directions, batch, hidden), quel que soit batch_first.
        # Bidirectionnel : h_n[-2] a lu la fenêtre vers l'avant, h_n[-1] vers l'arrière.
        last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.dropout(last)


class RecurrentRegressor(nn.Module):
    cell = "lstm"

    def __init__(self, n_features: int, **encoder_kwargs) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(n_features, cell=self.cell, **encoder_kwargs)
        self.head = nn.Linear(self.encoder.output_size, 1)

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(seq)).squeeze(-1)


class LSTMRegressor(RecurrentRegressor):
    cell = "lstm"


class GRURegressor(RecurrentRegressor):
    cell = "gru"


class MultiTaskRUL(nn.Module):
    """Deux entrées (séquence de capteurs, contexte de vol), deux sorties (RUL, état)."""

    def __init__(
        self,
        n_features: int,
        n_context: int,
        n_classes: int = 3,
        cell: str = "gru",
        use_context: bool = True,
        ctx_hidden: int = 16,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(n_features, cell=cell, **encoder_kwargs)
        self.use_context = use_context
        fused = self.encoder.output_size
        if use_context:
            self.ctx_encoder = nn.Sequential(nn.Linear(n_context, ctx_hidden), nn.ReLU())
            fused += ctx_hidden
        self.rul_head = nn.Linear(fused, 1)
        self.state_head = nn.Linear(fused, n_classes)  # logits bruts : pas de softmax

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(seq)
        if self.use_context:
            # dim=1 : on concatène le long des features, pas du batch (dim=0)
            features = torch.cat([features, self.ctx_encoder(ctx)], dim=1)
        return self.rul_head(features).squeeze(-1), self.state_head(features)


def build_model(cfg: dict, n_features: int, window: int, n_context: int) -> nn.Module:
    """Construit le modèle décrit par la section `model` d'une config."""
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if cfg["name"] == "mlp":
        return MLP(n_features, window, **kwargs)
    if cfg["name"] == "lstm":
        return LSTMRegressor(n_features, **kwargs)
    if cfg["name"] == "gru":
        return GRURegressor(n_features, **kwargs)
    if cfg["name"] == "multitask":
        return MultiTaskRUL(n_features, n_context, **kwargs)
    raise ValueError(f"Modèle inconnu : {cfg['name']}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
