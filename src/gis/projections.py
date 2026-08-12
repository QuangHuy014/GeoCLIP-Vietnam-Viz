# ==============================================================================
# MODULE: GIS Equal Earth Projection (GIS Analyst: Dev 2)
# RESPONSIBILITY: Project WGS84 (Lat, Lon) coordinates to 2D Equal Earth plane
# ==============================================================================

import torch

# Hằng số chuẩn của phép chiếu Equal Earth
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

def equal_earth_projection(L: torch.Tensor) -> torch.Tensor:
    """
    Chiếu tọa độ (Vĩ độ, Kinh độ) trên mặt cầu WGS84 sang mặt phẳng 2D Equal Earth.
    Args:
        L: Tensor (N, 2) với Cột 0 là Vĩ độ (Lat), Cột 1 là Kinh độ (Lon).
    Returns:
        Tensor (N, 2) tọa độ phẳng (x, y).
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    
    return (torch.stack((x, y), dim=1) * SF) / 180
