"""Modelos híbridos GCN+GRU y GraphSAGE+GRU para predicción espacio-temporal.

Combinan un encoder de grafos (GCN o GraphSAGE) con un GRU temporal:
1. El encoder de grafos procesa el grafo en cada paso temporal → embeddings
2. El GRU procesa la secuencia temporal de embeddings por nodo
3. El decoder MLP predice el tiempo de viaje por arista

Esto permite capturar tanto la estructura espacial del grafo como
los patrones temporales (horas punta, variación de demanda, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN
from torch_geometric.nn import SAGEConv


class _GraphGRUBase(nn.Module):
    """Clase base para modelos híbridos Graph+GRU.

    Subclases deben implementar `_build_graph_encoder` y `_graph_encode`.

    Args:
        in_channels: Dimensión de features de nodo.
        graph_hidden_dim: Dimensión oculta del encoder de grafos.
        graph_num_layers: Número de capas del encoder de grafos.
        gru_hidden_dim: Dimensión oculta del GRU.
        gru_num_layers: Número de capas GRU.
        edge_attr_dim: Dimensión de features de arista.
        dropout: Tasa de dropout.
    """

    def __init__(
        self,
        in_channels: int,
        graph_hidden_dim: int = 64,
        graph_num_layers: int = 2,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        edge_attr_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.graph_hidden_dim = graph_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.dropout = dropout

        # Subclases construyen el encoder de grafos
        self._build_graph_encoder(in_channels, graph_hidden_dim, graph_num_layers, dropout)

        # GRU temporal: procesa secuencia de embeddings de nodos
        self.gru = nn.GRU(
            input_size=graph_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        # Decoder MLP para predecir tiempo de viaje por arista
        decoder_in = gru_hidden_dim * 2 + edge_attr_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, gru_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden_dim // 2, 1),
        )

    def _build_graph_encoder(self, in_channels, hidden_dim, num_layers, dropout):
        raise NotImplementedError

    def _graph_encode(self, x, edge_index):
        raise NotImplementedError

    def forward(
        self,
        x_temporal: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass espacio-temporal.

        Args:
            x_temporal: Features temporales de nodos [N, T, F].
            edge_index: Conectividad del grafo [2, E].
            edge_attr: Features de aristas [E, edge_attr_dim].

        Returns:
            Predicciones de tiempo de viaje [E].
        """
        N, T, F = x_temporal.shape

        # 1. Encoder de grafos en cada paso temporal
        graph_embeddings = []
        for t in range(T):
            z_t = self._graph_encode(x_temporal[:, t, :], edge_index)  # [N, H_graph]
            graph_embeddings.append(z_t)

        # Stack: [N, T, H_graph]
        z_seq = torch.stack(graph_embeddings, dim=1)

        # 2. GRU temporal: procesa secuencia por nodo
        gru_out, _ = self.gru(z_seq)  # [N, T, H_gru]

        # Usar el último paso temporal como embedding final
        z_final = gru_out[:, -1, :]  # [N, H_gru]

        # 3. Decoder: predice tiempo por arista
        src, dst = edge_index
        edge_features = torch.cat([z_final[src], z_final[dst], edge_attr], dim=-1)
        return self.decoder(edge_features).squeeze(-1)


class TravelTimeGCN_GRU(_GraphGRUBase):
    """Modelo híbrido GCN + GRU.

    - GCN encoder captura la estructura espacial del grafo.
    - GRU captura patrones temporales (variación horaria de demanda/congestión).
    """

    def _build_graph_encoder(self, in_channels, hidden_dim, num_layers, dropout):
        self.graph_encoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
        )

    def _graph_encode(self, x, edge_index):
        return self.graph_encoder(x, edge_index)


class TravelTimeGraphSAGE_GRU(_GraphGRUBase):
    """Modelo híbrido GraphSAGE + GRU.

    - GraphSAGE encoder con SAGEConv captura la estructura espacial.
    - GRU captura patrones temporales.
    """

    def _build_graph_encoder(self, in_channels, hidden_dim, num_layers, dropout):
        self.sage_convs = nn.ModuleList()
        self.sage_norms = nn.ModuleList()

        self.sage_convs.append(SAGEConv(in_channels, hidden_dim))
        self.sage_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.sage_convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.sage_norms.append(nn.BatchNorm1d(hidden_dim))

    def _graph_encode(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.sage_convs, self.sage_norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            if i < len(self.sage_convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
