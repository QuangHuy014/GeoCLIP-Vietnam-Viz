# ==============================================================================
# MODULE: GIS Equal Earth Projection (GIS Analyst: Dev 2)
# RESPONSIBILITY: Project WGS84 (Lat, Lon) coordinates to 2D Equal Earth plane
# ==============================================================================

import torch
from src.gis.location_encoder import equal_earth_projection

__all__ = ["equal_earth_projection"]
