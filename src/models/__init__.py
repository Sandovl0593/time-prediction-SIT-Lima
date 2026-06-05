"""Models package — exporta todos los modelos disponibles."""

from src.models.graphsage_model import TravelTimeGraphSAGE
from src.models.gat_model import TravelTimeGAT
from src.models.gatv2_model import TravelTimeGATv2

__all__ = [
    "TravelTimeGraphSAGE",
    "TravelTimeGAT",
    "TravelTimeGATv2",
]
