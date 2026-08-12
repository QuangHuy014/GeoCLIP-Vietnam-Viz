# ==============================================================================
# MODULE: Location Encoder (Tech Lead: Dev 1)
# RESPONSIBILITY: Multi-scale Capsule Network for spatial location encoding
# ==============================================================================

import os
import torch
import torch.nn as nn
from src.gis.projections import equal_earth_projection
from src.gis.rff_layers import GaussianEncoding

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma: float):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.capsule(x)
        x = self.head(x)
        return x

class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], weights_path: str = None):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        if weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"[LocationEncoder] Loaded weights from: {weights_path}")

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], 512).to(location.device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        return location_features
