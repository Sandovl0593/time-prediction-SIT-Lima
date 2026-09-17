"""Models package — exporta los modelos disponibles en esta versión."""

from src.models.gat_model import TravelTimeGAT
from src.models.gatv2_model import TravelTimeGATv2

__all__ = [
    "TravelTimeGAT",
    "TravelTimeGATv2",
]
