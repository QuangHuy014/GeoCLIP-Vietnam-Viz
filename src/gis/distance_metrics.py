# ==============================================================================
# MODULE: GIS Distance Error Metrics (Đo đạc sai số khoảng cách & Chỉ số Acc@K)
# TRÁCH NHIỆM: Tính khoảng cách địa lý Geodesic / Haversine (km) & Đánh giá Acc@K
# ==============================================================================

import math
import os
import sys
from typing import List, Tuple, Dict, Union
import torch
import pandas as pd

# Thêm đường dẫn thư mục gốc vào sys.path để import khi chạy trực tiếp file
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False


# Bán kính trung bình của Trái Đất tính theo Kilômét
EARTH_RADIUS_KM = 6371.0088


def haversine_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Tính khoảng cách vòng lớn (great circle distance) giữa 2 điểm trên mặt cầu Trái Đất
    bằng công thức Haversine.

    Tham số:
        point1: Tuple (vĩ độ, kinh độ) của điểm thứ nhất (đơn vị: độ).
        point2: Tuple (vĩ độ, kinh độ) của điểm thứ hai (đơn vị: độ).
    Trả về:
        float: Khoảng cách giữa 2 điểm tính theo đơn vị Kilômét (km).
    """
    lat1, lon1 = point1
    lat2, lon2 = point2

    # Kiểm tra giới hạn tọa độ hợp lệ
    if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0):
        raise ValueError(f"Vĩ độ vượt quá khoảng hợp lệ [-90, 90]: {lat1}, {lat2}")
    if not (-180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0):
        raise ValueError(f"Kinh độ vượt quá khoảng hợp lệ [-180, 180]: {lon1}, {lon2}")

    # Đổi độ sang Radian
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Áp dụng công thức Haversine
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c


def calculate_geodesic_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    method: str = "haversine",
) -> float:
    """
    Tính khoảng cách thực tế chính xác giữa 2 tọa độ GPS tính bằng km.

    Tham số:
        point1 (Tuple[float, float]): (Vĩ độ, Kinh độ) điểm 1.
        point2 (Tuple[float, float]): (Vĩ độ, Kinh độ) điểm 2.
        method (str): Phương pháp tính ("haversine" hoặc "geodesic"). Mặc định: "haversine".
    Trả về:
        float: Khoảng cách địa lý tính bằng km.
    """
    if method == "geodesic" and HAS_GEOPY:
        return geodesic(point1, point2).km
    return haversine_distance(point1, point2)


def calculate_batch_haversine_distance(
    targets: Union[torch.Tensor, List[Tuple[float, float]]],
    predictions: Union[torch.Tensor, List[Tuple[float, float]]],
) -> List[float]:
    """
    Tính khoảng cách sai số (km) cho một tập hợp/mảng gồm nhiều cặp tọa độ thực tế và dự đoán.

    Tham số:
        targets: Danh sách hoặc Tensor danh sách tọa độ thực tế (N, 2).
        predictions: Danh sách hoặc Tensor danh sách tọa độ mô hình dự đoán (N, 2).
    Trả về:
        List[float]: Danh sách chứa sai số khoảng cách (km) của từng cặp điểm.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.tolist()

    if len(targets) != len(predictions):
        raise ValueError(f"Lỗi lệch kích thước: len(targets)={len(targets)} khác len(predictions)={len(predictions)}")

    distances = []
    for p1, p2 in zip(targets, predictions):
        dist = calculate_geodesic_distance((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        distances.append(dist)

    return distances


def compute_distance_accuracy_metrics(
    targets: List[Tuple[float, float]],
    predictions: List[Tuple[float, float]],
    thresholds_km: List[int] = [1, 25, 200, 750, 2500],
) -> Dict[str, float]:
    """
    Đánh giá tỷ lệ phần trăm chính xác theo các ngưỡng bán kính khoảng cách (Acc@1km, Acc@25km, Acc@200km...).

    Tham số:
        targets: Danh sách tọa độ thực tế [(lat, lon), ...].
        predictions: Danh sách tọa độ AI dự đoán [(lat, lon), ...].
        thresholds_km: Các ngưỡng bán kính khoảng cách tính bằng km.
    Trả về:
        Dict[str, float]: Từ điển chứa kết quả phần trăm độ chính xác và trung bình sai số.
    """
    if not targets or not predictions:
        raise ValueError("Danh sách targets và predictions không được để trống.")
    if len(targets) != len(predictions):
        raise ValueError(f"Độ dài không khớp: {len(targets)} targets vs {len(predictions)} predictions.")

    distances = calculate_batch_haversine_distance(targets, predictions)
    total_count = len(distances)

    metrics = {}
    for threshold in thresholds_km:
        correct_count = sum(1 for d in distances if d <= threshold)
        acc_pct = (correct_count / total_count) * 100.0
        metrics[f"Acc@{threshold}km"] = round(acc_pct, 2)

    metrics["mean_error_km"] = round(sum(distances) / total_count, 2)
    sorted_dists = sorted(distances)
    mid = total_count // 2
    if total_count % 2 == 0:
        median_val = (sorted_dists[mid - 1] + sorted_dists[mid]) / 2.0
    else:
        median_val = sorted_dists[mid]
    metrics["median_error_km"] = round(median_val, 2)

    return metrics


# ==============================================================================
# KHỐI CHẠY THỬ VÀ TRỰC QUAN HÓA KẾT QUẢ KHI CHẠY TRỰC TIẾP FILE NÀY
# ==============================================================================
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("======================================================================")
    print("       CHƯƠNG TRÌNH CHẠY THỬ MÔ-ĐUN ĐO ĐẠC KHOẢNG CÁCH SAI SỐ (GIS)    ")
    print("======================================================================\n")

    # --------------------------------------------------------------------------
    # MỤC 1: TEST TÍNH KHOẢNG CÁCH ĐƠN LẺ NỔI TIẾNG
    # --------------------------------------------------------------------------
    print("--- 📍 MỤC 1: Tính khoảng cách giữa các địa danh thực tế ---")
    hanoi = (21.0285, 105.8542)   # Tọa độ Hà Nội
    tphcm = (10.7769, 106.7009)   # Tọa độ TP. Hồ Chí Minh
    da_nang = (16.0544, 108.2022) # Tọa độ Đà Nẵng

    dist_hn_sg = calculate_geodesic_distance(hanoi, tphcm)
    dist_sg_dn = calculate_geodesic_distance(tphcm, da_nang)
    dist_same = calculate_geodesic_distance(tphcm, tphcm)

    print(f"1. Khoảng cách từ Hà Nội -> TP.HCM     : {dist_hn_sg:.2f} km")
    print(f"2. Khoảng cách từ TP.HCM -> Đà Nẵng    : {dist_sg_dn:.2f} km")
    print(f"3. Khoảng cách 2 điểm trùng (TP.HCM)  : {dist_same:.2f} km (Kỳ vọng: 0.00 km)")
    print()

    # --------------------------------------------------------------------------
    # MỤC 2: TEST TÍCH HỢP TẬP DỮ LIỆU THỰC TẾ TỪ FILE coordinates_100K.csv
    # --------------------------------------------------------------------------
    print("--- 📂 MỤC 2: Tích hợp ĐÁNH GIÁ DỮ LIỆU THỰC TẾ từ coordinates_100K.csv ---")
    csv_path = os.path.join(root_dir, "data", "coordinates_100K.csv")

    if os.path.exists(csv_path):
        df_real = pd.read_csv(csv_path)
        print(f"1. Đã đọc file thực tế: {os.path.basename(csv_path)} ({len(df_real):,} dòng)")

        # Lấy 500 vị trí thực tế làm ground-truth targets
        num_test = 500
        targets_real = list(zip(df_real['LAT'][:num_test], df_real['LON'][:num_test]))

        # Tạo giả lập độ lệch thực tế (nhiễu ngẫu nhiên bán kính nhỏ) đại diện cho dự đoán AI
        import random
        random.seed(42)
        preds_real = [
            (lat + random.uniform(-0.01, 0.01), lon + random.uniform(-0.01, 0.01))
            for lat, lon in targets_real
        ]

        print(f"2. Đo đạc sai số khoảng cách thực tế trên {num_test:,} mẫu...")
        metrics_real = compute_distance_accuracy_metrics(targets_real, preds_real)

        print("\n-----------------------------------------------------")
        print("  Chỉ số Đánh Giá Dữ Liệu Thực Tế   | Kết quả")
        print("-----------------------------------------------------")
        for key, val in metrics_real.items():
            unit = "%" if key.startswith("Acc@") else " km"
            print(f"  {key:<33} | {val:>8}{unit}")
        print("-----------------------------------------------------")
        print(f"✅ Đã đo đạc thành công sai số trên {num_test:,} vị trí THỰC TẾ!")
    else:
        print(f"⚠️ Không tìm thấy file {csv_path}")

    print("\n======================================================================")
    print("                HOÀN THÀNH CHẠY THỬ MÔ-ĐUN THÀNH CÔNG!                ")
    print("======================================================================")
