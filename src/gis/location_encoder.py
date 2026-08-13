# ==============================================================================
# MODULE: Standalone Location Encoder & Equal Earth Projection
# TRÁCH NHIỆM: Mã hóa tọa độ GPS WGS84 (Lat, Lon) thành Vector Embedding 512D
# ==============================================================================

import os
import sys
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import pandas as pd

# Thêm đường dẫn thư mục gốc vào sys.path để import khi chạy trực tiếp file
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.gis.rff_layers import MultiScaleGaussianEncoding


def equal_earth_projection(
    lat: Union[float, torch.Tensor], lon: Union[float, torch.Tensor] = None
) -> torch.Tensor:
    """
    Phép chiếu Equal Earth chuyển đổi tọa độ GPS WGS84 (Lat, Lon) sang mặt phẳng 2D (x, y).

    Có thể gọi theo 2 cách:
      - equal_earth_projection(lat, lon): truyền 2 số thực hoặc Tensor
      - equal_earth_projection(coords): truyền Tensor chứa cặp tọa độ (N, 2) hoặc (2,)

    Tham số:
        lat: Vĩ độ theo độ [-90, 90] hoặc Tensor tọa độ.
        lon: Kinh độ theo độ [-180, 180] (tùy chọn nếu lat đã là Tensor).
    Trả về:
        torch.Tensor: Tọa độ 2D đã chiếu (x, y) kích thước (N, 2) hoặc (2,).
    """
    if isinstance(lat, (list, tuple)):
        lat = torch.tensor(lat, dtype=torch.float32)

    if isinstance(lat, torch.Tensor) and lon is None:
        coords = lat
        if coords.dim() == 1:
            if coords.size(0) != 2:
                raise ValueError(f"Tensor tọa độ 1D phải có 2 phần tử (Lat, Lon), nhận được: {coords.size(0)}")
            lats = coords[0:1]
            lons = coords[1:2]
            is_1d = True
        elif coords.dim() == 2:
            if coords.size(1) != 2:
                raise ValueError(f"Tensor tọa độ 2D phải có dạng (N, 2), nhận được: {coords.shape}")
            lats = coords[:, 0]
            lons = coords[:, 1]
            is_1d = False
        else:
            raise ValueError(f"Tensor tọa độ phải là 1D hoặc 2D, nhận được: {coords.dim()}D")
    else:
        is_1d = not (isinstance(lat, torch.Tensor) and lat.dim() > 0)
        lats = torch.tensor([lat], dtype=torch.float32) if is_1d else torch.as_tensor(lat, dtype=torch.float32)
        lons = torch.tensor([lon], dtype=torch.float32) if is_1d else torch.as_tensor(lon, dtype=torch.float32)

    # Kiểm tra giới hạn tọa độ [-90, 90] và [-180, 180]
    if torch.any(lats < -90.0) or torch.any(lats > 90.0):
        raise ValueError(f"Vĩ độ vượt quá khoảng hợp lệ [-90, 90]: {lats.tolist()}")
    if torch.any(lons < -180.0) or torch.any(lons > 180.0):
        raise ValueError(f"Kinh độ vượt quá khoảng hợp lệ [-180, 180]: {lons.tolist()}")

    # Đổi độ sang Radian
    lat_rad = torch.deg2rad(lats)
    lon_rad = torch.deg2rad(lons)

    # Vĩ độ tham số (psi)
    sin_psi = (3.0 ** 0.5 / 2.0) * torch.sin(lat_rad)
    psi = torch.asin(torch.clamp(sin_psi, -1.0, 1.0))

    psi2 = psi ** 2
    psi4 = psi2 ** 2
    psi6 = psi4 * psi2
    psi8 = psi4 ** 2

    # Các hệ số đa thức phép chiếu Equal Earth
    a0 = 1.340264
    a1 = -0.081106
    a2 = 0.000893
    a3 = 0.003796
    a4 = 0.003429

    denom = 3.0 * (9.0 * a4 * psi8 + 7.0 * a3 * psi6 + 5.0 * a2 * psi4 + 3.0 * a1 * psi2 + a0)
    x = (2.0 * (3.0 ** 0.5) * lon_rad * torch.cos(psi)) / denom
    y = psi * (a4 * psi8 + a3 * psi6 + a2 * psi4 + a1 * psi2 + a0)

    result = torch.stack([x, y], dim=-1)

    # Kiểm tra đảm bảo không có NaN hay Inf
    if torch.isnan(result).any() or torch.isinf(result).any():
        raise RuntimeError("Kết quả phép chiếu Equal Earth chứa giá trị NaN hoặc Inf")

    if is_1d:
        result = result.squeeze(0)

    return result


class StandaloneLocationEncoder(nn.Module):
    """
    Bộ mã hóa vị trí độc lập (Standalone Location Encoder) chuyển đổi GPS (Lat, Lon) thành Vector 512D.
    Luồng xử lý (Pipeline):
      1. Equal Earth Projection: Tọa độ GPS (Lat, Lon) -> Mặt phẳng 2D (x, y)
      2. Multi-scale RFF: Tọa độ 2D (x, y) -> Chuỗi đặc trưng sóng đa tần số (1536D)
      3. MLP Projection: Mạng Neural chiếu 1536D -> 1024D -> 512D
    """

    def __init__(
        self,
        sigmas: tuple = (1.0, 16.0, 256.0),
        encoded_size_per_scale: int = 256,
        embed_dim: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        self.rff_encoder = MultiScaleGaussianEncoding(
            sigmas=list(sigmas),
            input_size=2,
            encoded_size_per_scale=encoded_size_per_scale,
            seed=seed,
        )

        in_dim = self.rff_encoder.out_dim  # 3 dải * (2 * 256) = 1536
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim),
        )

    def load_pretrained_weights(self, weights_path: str = "location_encoder_weights.pth") -> bool:
        """
        Nạp trọng số mô hình pre-trained từ tệp nếu có.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state_dict)
        return True

    def forward(
        self,
        L: Union[torch.Tensor, list, Tuple[float, float]],
        lon: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Hàm lan truyền tiến (Forward Pass).

        Tham số:
            L: Tensor dạng (N, 2) hoặc (2,), hoặc list/tuple tọa độ GPS.
            lon: Kinh độ float (nếu L là vĩ độ float).
        Trả về:
            torch.Tensor: Vector Location Embedding dạng (N, 512) hoặc (1, 512).
        """
        # 1. Phép chiếu Equal Earth
        if lon is not None:
            v_2d = equal_earth_projection(L, lon)
        else:
            v_2d = equal_earth_projection(L)

        if v_2d.dim() == 1:
            v_2d = v_2d.unsqueeze(0)

        # 2. Mã hóa tần số RFF
        rff_feats = self.rff_encoder(v_2d)

        # 3. Mạng MLP chiếu sang 512 chiều
        embeddings = self.mlp(rff_feats)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            raise RuntimeError("Kết quả Location Encoder chứa giá trị NaN hoặc Inf")

        return embeddings


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("======================================================================")
    print("      CHƯƠNG TRÌNH CHẠY THỬ STANDALONE LOCATION ENCODER (512D)         ")
    print("======================================================================\n")

    encoder = StandaloneLocationEncoder()

    # --------------------------------------------------------------------------
    # MỤC 1: TEST ĐƠN TỌA ĐỘ VỚI TỌA ĐỘ MẪU
    # --------------------------------------------------------------------------
    print("--- 📍 MỤC 1: Test mã hóa tọa độ mẫu (TP.HCM & Hà Nội) ---")
    gps_hcm = [[10.795, 106.721]]
    emb_hcm = encoder(gps_hcm)
    print(f"1. TP.HCM [10.795, 106.721] -> Embedding Shape: {emb_hcm.shape}")
    print(f"   5 Giá trị đầu: {emb_hcm[0, :5].detach().numpy()}")

    gps_hn = [[21.028, 105.834]]
    emb_hn = encoder(gps_hn)
    print(f"2. Hà Nội [21.028, 105.834] -> Embedding Shape: {emb_hn.shape}")
    print(f"   5 Giá trị đầu: {emb_hn[0, :5].detach().numpy()}\n")

    # --------------------------------------------------------------------------
    # MỤC 2: TÍCH HỢP TÍNH TOÁN DỮ LIỆU THỰC TẾ TỪ FILE coordinates_100K.csv
    # --------------------------------------------------------------------------
    print("--- 📂 MỤC 2: Tích hợp nạp DỮ LIỆU THỰC TẾ từ coordinates_100K.csv ---")
    csv_path = os.path.join(root_dir, "data", "coordinates_100K.csv")

    if os.path.exists(csv_path):
        print(f"1. Đọc dữ liệu từ file thực tế: {os.path.basename(csv_path)}...")
        df = pd.read_csv(csv_path)
        print(f"   -> Tổng số dòng tọa độ thực tế tìm thấy: {len(df):,} dòng")

        # Nạp 1,000 tọa độ thực tế đầu tiên làm dữ liệu mẫu kiểm thử
        num_sample = 1000
        real_coords_tensor = torch.tensor(df[['LAT', 'LON']].values[:num_sample], dtype=torch.float32)
        
        print(f"2. Mã hóa hàng loạt {num_sample:,} tọa độ thật qua StandaloneLocationEncoder...")
        with torch.no_grad():
            real_embeddings = encoder(real_coords_tensor)

        print(f"   -> Input Shape  : {real_coords_tensor.shape} (1,000 tọa độ [Lat, Lon] thực tế)")
        print(f"   -> Output Shape : {real_embeddings.shape} (Kỳ vọng: [1000, 512])")
        print(f"   -> Không có NaN/Inf: {not torch.isnan(real_embeddings).any()}")
        print(f"   -> Giá trị Min/Max Embedding: {real_embeddings.min().item():.4f} / {real_embeddings.max().item():.4f}")
        print(f"✅ Đã mã hóa thành công {num_sample:,} tọa độ THỰC TẾ từ coordinates_100K.csv!")
    else:
        print(f"⚠️ Không tìm thấy file {csv_path}")

    print("\n======================================================================")
    print("                HOÀN THÀNH CHẠY THỬ MÔ-ĐUN THÀNH CÔNG!                ")
    print("======================================================================")
