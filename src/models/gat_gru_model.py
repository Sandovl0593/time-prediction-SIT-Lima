"""Modelos híbridos GAT+GRU y GATv2+GRU para predicción espacio-temporal.

Combinan un encoder de grafos basado en atención (GAT o GATv2) con un
GRU temporal:
1. El encoder de atención procesa el grafo en cada paso temporal → embeddings
2. El GRU procesa la secuencia temporal de embeddings por nodo
3. El decoder MLP predice el tiempo de viaje por arista

Los mecanismos de atención permiten al modelo aprender qué vecinos son
más relevantes para cada nodo, complementando la captura de patrones
temporales del GRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv

from src.models.gru_model import _GraphGRUBase


class TravelTimeGAT_GRU(_GraphGRUBase):
    """Modelo híbrido GAT + GRU.

    - GAT encoder con multi-head attention captura la estructura espacial
      del grafo, aprendiendo la importancia relativa de cada vecino.
    - GRU captura patrones temporales (variación horaria de demanda/congestión).

    Args:
        in_channels: Dimensión de features de nodo.
        graph_hidden_dim: Dimensión oculta del encoder GAT (por head).
        graph_num_layers: Número de capas GATConv.
        heads: Número de heads de atención.
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
        heads: int = 4,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        edge_attr_dim: int = 2,
        dropout: float = 0.1,
    ):
        self._heads = heads
        super().__init__(
            in_channels=in_channels,
            graph_hidden_dim=graph_hidden_dim,
            graph_num_layers=graph_num_layers,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
        )

    def _build_graph_encoder(self, in_channels, hidden_dim, num_layers, dropout):
        heads = self._heads
        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # Primera capa
        self.gat_convs.append(
            GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        )
        self.gat_norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.gat_convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
            self.gat_norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Última capa: promedio de heads → hidden_dim
        if num_layers > 1:
            self.gat_convs.append(
                GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )
            self.gat_norms.append(nn.BatchNorm1d(hidden_dim))

    def _graph_encode(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.gat_convs, self.gat_norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            if i < len(self.gat_convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TravelTimeGATv2_GRU(_GraphGRUBase):
    """Modelo híbrido GATv2 + GRU.

    - GATv2 encoder con atención dinámica captura la estructura espacial,
      superando la limitación de atención estática del GAT original.
    - GRU captura patrones temporales.

    Args:
        in_channels: Dimensión de features de nodo.
        graph_hidden_dim: Dimensión oculta del encoder GATv2 (por head).
        graph_num_layers: Número de capas GATv2Conv.
        heads: Número de heads de atención.
        gru_hidden_dim: Dimensión oculta del GRU.
        gru_num_layers: Número de capas GRU.
        edge_attr_dim: Dimensión de features de arista.
        dropout: Tasa de dropout.
        share_weights: Si True, comparte pesos entre transformaciones.
    """

    def __init__(
        self,
        in_channels: int,
        graph_hidden_dim: int = 64,
        graph_num_layers: int = 2,
        heads: int = 4,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        edge_attr_dim: int = 2,
        dropout: float = 0.1,
        share_weights: bool = False,
    ):
        self._heads = heads
        self._share_weights = share_weights
        super().__init__(
            in_channels=in_channels,
            graph_hidden_dim=graph_hidden_dim,
            graph_num_layers=graph_num_layers,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
        )

    def _build_graph_encoder(self, in_channels, hidden_dim, num_layers, dropout):
        heads = self._heads
        share_weights = self._share_weights
        self.gatv2_convs = nn.ModuleList()
        self.gatv2_norms = nn.ModuleList()

        # Primera capa
        self.gatv2_convs.append(
            GATv2Conv(
                in_channels, hidden_dim, heads=heads,
                dropout=dropout, share_weights=share_weights,
            )
        )
        self.gatv2_norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.gatv2_convs.append(
                GATv2Conv(
                    hidden_dim * heads, hidden_dim, heads=heads,
                    dropout=dropout, share_weights=share_weights,
                )
            )
            self.gatv2_norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Última capa: promedio de heads → hidden_dim
        if num_layers > 1:
            self.gatv2_convs.append(
                GATv2Conv(
                    hidden_dim * heads, hidden_dim, heads=1,
                    concat=False, dropout=dropout, share_weights=share_weights,
                )
            )
            self.gatv2_norms.append(nn.BatchNorm1d(hidden_dim))

    def _graph_encode(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.gatv2_convs, self.gatv2_norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            if i < len(self.gatv2_convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
