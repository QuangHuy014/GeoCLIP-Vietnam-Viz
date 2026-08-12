import json
import os
import torch
import pandas as pd
import numpy as np

geojson_path = "data/hotosm_vnm_points_of_interest_points_geojson.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    data = json.load(f)

points = data.get("features", [])
l81_points = []
for p in points:
    geom = p.get("geometry", {})
    if geom.get("type") == "Point":
        coords = geom.get("coordinates", [])
        if len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            # Landmark 81: Lat = 10.795, Lon = 106.721
            if 10.78 <= lat <= 10.81 and 106.71 <= lon <= 106.73:
                name = p.get("properties", {}).get("name")
                l81_points.append((name, lat, lon))

print(f"Total points near Landmark 81 (radius ~1-2km) in GeoJSON: {len(l81_points)}")
for name, lat, lon in l81_points[:10]:
    print(f" - Name: {name} | Lat: {lat:.6f}, Lon: {lon:.6f}")
