# ==============================================================================
# MODULE: GIS Equal Earth Projection (Phép chiếu Equal Earth)
# TRÁCH NHIỆM: Chiếu tọa độ cầu WGS84 (Lat, Lon) sang mặt phẳng 2D Equal Earth
# ==============================================================================

import sys
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

    Tham số:
        L: Tensor kích thước (N, 2) hoặc (2,) với Cột 0 là Vĩ độ (Lat), Cột 1 là Kinh độ (Lon).
    Trả về:
        Tensor kích thước (N, 2) hoặc (2,) chứa tọa độ phẳng (x, y).
    """
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, dtype=torch.float32)

    is_1d = L.dim() == 1
    if is_1d:
        L = L.unsqueeze(0)

    latitude = L[:, 0]
    longitude = L[:, 1]

    # Kiểm tra khoảng hợp lệ
    if torch.any(latitude < -90.0) or torch.any(latitude > 90.0):
        raise ValueError(f"Vĩ độ vượt quá khoảng hợp lệ [-90, 90]: {latitude}")
    if torch.any(longitude < -180.0) or torch.any(longitude > 180.0):
        raise ValueError(f"Kinh độ vượt quá khoảng hợp lệ [-180, 180]: {longitude}")

    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)

    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta

    result = (torch.stack((x, y), dim=1) * SF) / 180

    if is_1d:
        result = result.squeeze(0)

    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("======================================================================")
    print("      CHƯƠNG TRÌNH CHẠY THỬ PHÉP CHIẾU EQUAL EARTH PROJECTION         ")
    print("======================================================================\n")

    # 1. Test TP.HCM
    gps_hcm = torch.tensor([10.795, 106.721])
    xy_hcm = equal_earth_projection(gps_hcm)
    print(f"📍 TP.HCM (Lat: 10.795, Lon: 106.721) -> Tọa độ 2D (x, y): {xy_hcm.numpy()}")

    # 2. Test Hà Nội
    gps_hn = torch.tensor([21.028, 105.834])
    xy_hn = equal_earth_projection(gps_hn)
    print(f"📍 Hà Nội (Lat: 21.028, Lon: 105.834) -> Tọa độ 2D (x, y): {xy_hn.numpy()}")

    # 3. Test Batch nhiều địa danh
    batch_gps = torch.tensor([
        [10.795, 106.721], # TP.HCM
        [21.028, 105.834], # Hà Nội
        [16.054, 108.202], # Đà Nẵng
    ])
    batch_xy = equal_earth_projection(batch_gps)
    print(f"\n🔹 Input Batch Shape : {batch_gps.shape}")
    print(f"🔹 Output Batch Shape: {batch_xy.shape}")
    print("======================================================================")
