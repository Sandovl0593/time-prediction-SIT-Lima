"""GCN para predicción de tiempos de viaje sobre aristas.

Utiliza torch_geometric.nn.models.GCN como encoder de nodos, luego
concatena embeddings de pares (src, dst) junto con features de arista
para predecir el tiempo de viaje.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models import GCN


class TravelTimeGCN(nn.Module):
    """GCN encoder + edge-level MLP decoder para regresión de tiempos.

    Arquitectura:
        1. GCN encoder: in_channels → hidden_dim (num_layers capas)
        2. Edge decoder MLP: [z_src ‖ z_dst ‖ edge_attr] → 1

    Args:
        in_channels: Dimensión de features de nodo.
        hidden_dim: Dimensión oculta del GCN.
        num_layers: Número de capas GCNConv.
        edge_attr_dim: Dimensión de features de arista.
        dropout: Tasa de dropout.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        edge_attr_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
        )

        # Decoder MLP para predecir tiempo de viaje por arista
        decoder_in = hidden_dim * 2 + edge_attr_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encoder: obtiene embeddings de nodos."""
        return self.encoder(x, edge_index)

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder: predice tiempos de viaje por arista."""
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst], edge_attr], dim=-1)
        return self.decoder(edge_features).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass completo: encode → decode.

        Args:
            x: Features de nodos [N, in_channels].
            edge_index: Conectividad del grafo [2, E].
            edge_attr: Features de aristas [E, edge_attr_dim].

        Returns:
            Predicciones de tiempo de viaje [E].
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index, edge_attr)
