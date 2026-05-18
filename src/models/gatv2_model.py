"""GATv2 para predicción de tiempos de viaje sobre aristas.

Utiliza un stack de GATv2Conv (Graph Attention Network v2) como encoder
de nodos, luego concatena embeddings de pares (src, dst) junto con
features de arista para predecir el tiempo de viaje.

GATv2 corrige la limitación de "atención estática" de GAT original
aplicando la no-linealidad ANTES de calcular los coeficientes de
atención, lo que permite atención dinámica que depende de los features
del nodo consulta.

Referencia: Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class TravelTimeGATv2(nn.Module):
    """GATv2 encoder + edge-level MLP decoder para regresión de tiempos.

    Arquitectura:
        1. Stack de GATv2Conv: in_channels → hidden_dim (num_layers capas)
           con multi-head attention dinámica (los heads se concatenan en
           capas intermedias y se promedian en la última capa).
        2. Edge decoder MLP: [z_src ‖ z_dst ‖ edge_attr] → 1

    Args:
        in_channels: Dimensión de features de nodo.
        hidden_dim: Dimensión oculta por head de atención.
        num_layers: Número de capas GATv2Conv.
        heads: Número de heads de atención por capa.
        edge_attr_dim: Dimensión de features de arista.
        dropout: Tasa de dropout (aplicado a features y coeficientes de atención).
        share_weights: Si True, comparte pesos entre la transformación de
            nodo fuente y destino (parámetro exclusivo de GATv2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        edge_attr_dim: int = 2,
        dropout: float = 0.1,
        share_weights: bool = False,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        # Primera capa: in_channels → hidden_dim * heads (concatenación)
        self.convs.append(
            GATv2Conv(
                in_channels,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                share_weights=share_weights,
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Capas intermedias: hidden_dim * heads → hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    share_weights=share_weights,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Última capa: hidden_dim * heads → hidden_dim (promedio de heads)
        if num_layers > 1:
            self.convs.append(
                GATv2Conv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    share_weights=share_weights,
                )
            )
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
        """Encoder: stack de GATv2Conv con BatchNorm, ELU y dropout."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
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
