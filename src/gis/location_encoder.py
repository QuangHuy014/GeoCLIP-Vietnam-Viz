# ==============================================================================
# MODULE: Standalone Location Encoder & Equal Earth Projection
# RESPONSIBILITY: Converts WGS84 GPS (Lat, Lon) to 512-dim Location Embedding
# ==============================================================================

import os
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn

from src.gis.rff_layers import MultiScaleGaussianEncoding

#Tạo một function nhận vĩ độ và kinh độ.
def equal_earth_projection(
    lat: Union[float, torch.Tensor],
    lon: Union[float, torch.Tensor] = None
) -> torch.Tensor:
    """
    Equal Earth Projection transforming WGS84 (Lat, Lon) in degrees to 2D flat (x, y) coordinates.

    Can be called as:
      - equal_earth_projection(lat, lon) where lat, lon are floats or Tensors
      - equal_earth_projection(coords) where coords is a Tensor of shape (N, 2) or (2,)

    Args:
        lat: Latitude in degrees [-90, 90] or coordinate Tensor.
        lon: Longitude in degrees [-180, 180] (optional if coords tensor passed in lat).

    Returns:
        torch.Tensor: Projected 2D coordinates (x, y) of shape (N, 2) or (2,).
    """
    #Kiểm tra biến có thuộc kiểu nào không
    #lat có phải list hoặc tuple không
    if isinstance(lat, (list, tuple)):
        lat = torch.tensor(lat, dtype=torch.float32)

    #Nếu lat là Tensor và người dùng không truyền lon riêng.
    if isinstance(lat, torch.Tensor) and lon is None:
        coords = lat
        #dim() trả về số chiều của Tensor
        if coords.dim() == 1:
        #Trả về 2 phần tử không được nhỏ hơn hoặc lớn hơn
            if coords.size(0) != 2:
                raise ValueError(f"Tạo độ 1D phải có kích thước 2, có {coords.size(0)}")
        #Lấy vĩ độ và kinh độ
            lats = coords[0:1]
            lons = coords[1:2]
        #Ban đầu người dùng đưa vào một tọa độ hay một batch
            is_1d = True
        elif coords.dim() == 2:
            if coords.size(1) != 2:
                raise ValueError(f"Tọa độ 2D phải có hình dạng (N, 2), có {coords.shape}")
            lats = coords[:, 0]
            lons = coords[:, 1]
            is_1d = False
        else:
            raise ValueError(f"Tọa độ phải là 1D hoặc 2D, có {coords.dim()}D")
    else:
        is_1d = not (isinstance(lat, torch.Tensor) and lat.dim() > 0)
        lats = torch.tensor([lat], dtype=torch.float32) if is_1d else torch.as_tensor(lat, dtype=torch.float32)
        lons = torch.tensor([lon], dtype=torch.float32) if is_1d else torch.as_tensor(lon, dtype=torch.float32)

    # Trả về thông báo không hợp lệ
    if torch.any(lats < -90.0) or torch.any(lats > 90.0):
        raise ValueError(f"Vĩ độ ngoài phạm vi hợp lệ [-90, 90]: {lats}")
    if torch.any(lons < -180.0) or torch.any(lons > 180.0):
        raise ValueError(f"Kinh độ ngoài phạm vi hợp lệ [-180, 180]: {lons}")

    # Chuyển degrees sang radians
    lat_rad = torch.deg2rad(lats)
    lon_rad = torch.deg2rad(lons)

    # Tính vĩ độ
    #Hàm asin() chỉ chấp nhận:-1 ≤ x ≤ 1
    sin_psi = (3.0 ** 0.5 / 2.0) * torch.sin(lat_rad)
    psi = torch.asin(torch.clamp(sin_psi, -1.0, 1.0))

    #Phục vụ công thức Equal Earth chuyển thành số mũ
    psi2 = psi ** 2
    psi4 = psi2 ** 2
    psi6 = psi4 * psi2
    psi8 = psi4 ** 2

    # Hệ số đa thức Trái đất bằng nhau
    a0 = 1.340264
    a1 = -0.081106
    a2 = 0.000893
    a3 = 0.003796
    a4 = 0.003429

    #mẫu số trong công thức projection.
    denom = 3.0 * (9.0 * a4 * psi8 + 7.0 * a3 * psi6 + 5.0 * a2 * psi4 + 3.0 * a1 * psi2 + a0)
    #Tính kinh độ
    x = (2.0 * (3.0 ** 0.5) * lon_rad * torch.cos(psi)) / denom
    #Tính vĩ độ
    y = psi * (a4 * psi8 + a3 * psi6 + a2 * psi4 + a1 * psi2 + a0)

    result = torch.stack([x, y], dim=-1)

    # Đầu ra không NaN hoặc Inf
    if torch.isnan(result).any() or torch.isinf(result).any():
        raise RuntimeError("Equal Earth Projection output contains NaN or Inf values")

    if is_1d:
        result = result.squeeze(0)

    return result


class StandaloneLocationEncoder(nn.Module):
    """
    Standalone Location Encoder mapping GPS coordinates (Lat, Lon) to 512-dim Embeddings.
    Pipeline:
      1. Equal Earth Projection: (Lat, Lon) -> (x, y)
      2. Multi-scale RFF: (x, y) -> Multi-frequency Fourier features (dim=1536)
      3. MLP Projection: 1536 -> 1024 -> 512
    """

    def __init__(
        self,
        sigmas: tuple = (1.0, 16.0, 256.0),
        encoded_size_per_scale: int = 256,
        embed_dim: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        # Tạo RFF encoder.
        self.rff_encoder = MultiScaleGaussianEncoding(
            sigmas=list(sigmas),
            input_size=2,
            encoded_size_per_scale=encoded_size_per_scale,
            seed=seed,
        )

        in_dim = self.rff_encoder.out_dim  # 3 * (2 * 256) = 1536
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim),
        )

    def load_pretrained_weights(self, weights_path: str = "location_encoder_weights.pth") -> bool:
        """
        Loads pre-trained weights from file if available.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weight file not found at: {weights_path}")
        
        state_dict = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state_dict)
        return True

    def forward(
        self,
        L: Union[torch.Tensor, list, Tuple[float, float]],
        lon: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass for location encoder.

        Args:
            L: Tensor of shape (N, 2) or (2,), or list/tuple of coordinates.
            lon: Optional longitude float if L is a latitude float.

        Returns:
            torch.Tensor: Location embedding of shape (N, 512) or (1, 512).
        """
        # 1. Equal Earth Projection
        if lon is not None:
            v_2d = equal_earth_projection(L, lon)
        else:
            v_2d = equal_earth_projection(L)

        if v_2d.dim() == 1:
            v_2d = v_2d.unsqueeze(0)

        # 2. Multi-scale RFF Encoding
        rff_feats = self.rff_encoder(v_2d)

        # 3. MLP Projection to 512 dimensions
        embeddings = self.mlp(rff_feats)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            raise RuntimeError("Location Encoder output contains NaN or Inf values")

        return embeddings
