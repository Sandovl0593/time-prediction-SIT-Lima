"""GraphSAGE para predicción de tiempos de viaje sobre aristas.

Utiliza un stack de SAGEConv como encoder de nodos, luego concatena
embeddings de pares (src, dst) junto con features de arista para
predecir el tiempo de viaje.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TravelTimeGraphSAGE(nn.Module):
    """GraphSAGE encoder + edge-level MLP decoder para regresión de tiempos.

    Arquitectura:
        1. Stack de SAGEConv: in_channels → hidden_dim (num_layers capas)
        2. Edge decoder MLP: [z_src ‖ z_dst ‖ edge_attr] → 1

    Args:
        in_channels: Dimensión de features de nodo.
        hidden_dim: Dimensión oculta.
        num_layers: Número de capas SAGEConv.
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        # Primera capa
        self.convs.append(SAGEConv(in_channels, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Capas intermedias
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

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
        """Encoder: stack de SAGEConv con BatchNorm y ReLU."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

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
