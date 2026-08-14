# ==============================================================================
# GeoCLIP Vietnam Library Entrypoint
# ==============================================================================

from src.core.pipeline import GeoCLIPService
from src.core.image_encoder import StandaloneImageEncoder
from src.core.matcher import CosineMatcher
from src.core.location_encoder import LocationEncoder
from src.gis.distance_metrics import (
    calculate_geodesic_distance,
    calculate_batch_haversine_distance,
    compute_distance_accuracy_metrics,
    haversine_distance
)
from src.viz.map_builder import create_prediction_map

__version__ = "1.0.0"
__author__ = "UIT Data Analytics Team"

__all__ = [
    "GeoCLIPService",
    "StandaloneImageEncoder",
    "LocationEncoder",
    "CosineMatcher",
    "calculate_geodesic_distance",
    "calculate_batch_haversine_distance",
    "compute_distance_accuracy_metrics",
    "haversine_distance",
    "create_prediction_map"
]
