"""Models package — exporta todos los modelos disponibles."""

from src.models.gcn_model import TravelTimeGCN
from src.models.graphsage_model import TravelTimeGraphSAGE
from src.models.gat_model import TravelTimeGAT
from src.models.gatv2_model import TravelTimeGATv2
# from src.models.gru_model import TravelTimeGCN_GRU, TravelTimeGraphSAGE_GRU
from src.models.gat_gru_model import TravelTimeGAT_GRU, TravelTimeGATv2_GRU

__all__ = [
    "TravelTimeGCN",
    "TravelTimeGraphSAGE",
    "TravelTimeGAT",
    "TravelTimeGATv2",
    # "TravelTimeGCN_GRU",
    # "TravelTimeGraphSAGE_GRU",
    "TravelTimeGAT_GRU",
    "TravelTimeGATv2_GRU",
]
