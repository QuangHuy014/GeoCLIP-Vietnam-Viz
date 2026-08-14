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

    metrics["Mean_Error_km"] = round(sum(distances) / total_count, 2)
    sorted_dists = sorted(distances)
    mid = total_count // 2
    if total_count % 2 == 0:
        median_val = (sorted_dists[mid - 1] + sorted_dists[mid]) / 2.0
    else:
        median_val = sorted_dists[mid]
    metrics["Median_Error_km"] = round(median_val, 2)

    return metrics

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("======================================================================")
    print("       CHƯƠNG TRÌNH CHẠY THỬ MÔ-ĐUN ĐO ĐẠC KHOẢNG CÁCH SAI SỐ (GIS)    ")
    print("======================================================================\n")

    hanoi = (21.0285, 105.8542)
    tphcm = (10.7769, 106.7009)
    dist = calculate_geodesic_distance(hanoi, tphcm)
    print(f"Khoảng cách Hà Nội -> TP.HCM: {dist:.2f} km")
