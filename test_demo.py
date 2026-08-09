import torch

from src.gis.location_encoder import (
    equal_earth_projection,
    StandaloneLocationEncoder
)
from src.gis.rff_layers import MultiScaleGaussianEncoding


# ==================================================
# 1. Equal Earth
# ==================================================

print("=== 1. EQUAL EARTH ===")

hcm_xy = equal_earth_projection(10.795, 106.721)
hanoi_xy = equal_earth_projection(21.028, 105.834)

print("HCM   :", hcm_xy)
print("Hanoi :", hanoi_xy)

try:
    equal_earth_projection(95.0, 106.721)
except ValueError as e:
    print("Invalid latitude -> OK:", e)


# ==================================================
# 2. RFF
# ==================================================

print("\n=== 2. RFF ===")

rff = MultiScaleGaussianEncoding(
    sigmas=[1.0, 16.0, 256.0]
)

xy = torch.tensor([[0.5, 0.5]])

rff_out = rff(xy)
rff_out2 = rff(xy)

print("Input shape :", xy.shape)
print("Output shape:", rff_out.shape)
print("Deterministic:", torch.equal(rff_out, rff_out2))


# ==================================================
# 3. Location Encoder
# ==================================================

print("\n=== 3. LOCATION ENCODER ===")

encoder = StandaloneLocationEncoder()

hcm = torch.tensor([
    [10.795, 106.721]
])

embedding = encoder(hcm)

print("Input shape :", hcm.shape)
print("Output shape:", embedding.shape)
print("First 5 values:", embedding[0, :5])


# ==================================================
# 4. Batch
# ==================================================

print("\n=== 4. BATCH ===")

landmarks = torch.tensor([
    [10.795, 106.721],
    [21.028, 105.834],
    [16.054, 108.202]
])

embeddings = encoder(landmarks)

print("Input shape :", landmarks.shape)
print("Output shape:", embeddings.shape)

print(
    "Contains NaN/Inf:",
    not torch.isfinite(embeddings).all()
)