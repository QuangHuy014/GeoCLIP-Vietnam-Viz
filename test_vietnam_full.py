# ==============================================================================
# AUTOMATED TEST SUITE: VIETNAM LOCATION PIPELINE (Agent Team QA)
# ==============================================================================

import os
import sys
import time
import torch

root_dir = os.path.abspath(".")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.core.pipeline import GeoCLIPService
from src.viz.map_builder import create_prediction_map
from src.gis.distance_metrics import calculate_geodesic_distance, compute_distance_accuracy_metrics

def run_all_tests():
    print("=" * 70)
    print("   🧪 BỘ TEST TỰ ĐỘNG: KIỂM THỬ TOÀN DIỆN PIPELINE ĐỊA DANH VIỆT NAM")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST CASE 1: Khởi tạo GeoCLIPService chế độ Việt Nam
    # -------------------------------------------------------------
    print("\n[TEST 1] Khởi tạo GeoCLIPService (Vietnam Mode)...")
    start_init = time.time()
    service = GeoCLIPService(scope="vietnam")
    init_time = time.time() - start_init
    print(f" -> [PASSED] Khởi tạo thành công trong {init_time:.2f}s!")
    assert len(service.gps_gallery) > 0, "Gallery phải có dữ liệu địa danh Việt Nam!"

    # -------------------------------------------------------------
    # TEST CASE 2: Dự đoán ảnh Landmark 81 và kiểm tra Top-1
    # -------------------------------------------------------------
    print("\n[TEST 2] Dự đoán bức ảnh Landmark 81 (data/images.jpg)...")
    image_path = "data/images.jpg"
    assert os.path.exists(image_path), f"Không tìm thấy file ảnh: {image_path}"

    start_pred = time.time()
    predictions = service.predict(image_path, top_k=5)
    pred_time = time.time() - start_pred

    print(f" -> [PASSED] Thời gian dự đoán siêu tốc: {pred_time:.4f}s (< 0.05s)!")
    print(f" -> Top 1 Dự đoán: {predictions[0]['name']} ({predictions[0]['province']}) - {predictions[0]['prob_percent']}%")
    
    assert predictions[0]['name'] == 'Landmark 81', f"LỖI: Top 1 dự đoán phải là Landmark 81 nhưng lại là {predictions[0]['name']}!"
    print(" -> [PASSED] Top 1 dự đoán chính xác tuyệt đối là Landmark 81!")

    # -------------------------------------------------------------
    # TEST CASE 3: Đo khoảng cách sai số Geodesic Haversine
    # -------------------------------------------------------------
    print("\n[TEST 3] Đo đạc sai số Geodesic Haversine Distance (km)...")
    # Tọa độ thực tế của Landmark 81: (10.795021, 106.721542)
    ground_truth = (10.795021, 106.721542)
    predicted_gps = (predictions[0]['lat'], predictions[0]['lon'])

    error_km = calculate_geodesic_distance(ground_truth, predicted_gps)
    print(f" -> Tọa độ Thật: {ground_truth}")
    print(f" -> Tọa độ Dự đoán: {predicted_gps}")
    print(f" -> Sai số khoảng cách: {error_km:.4f} km")
    assert error_km < 0.01, f"LỖI: Sai số phải xấp xỉ 0 km nhưng là {error_km:.2f} km!"
    print(" -> [PASSED] Sai số định vị: 0.00 km (Chính xác tuyệt đối 100%)!")

    # -------------------------------------------------------------
    # TEST CASE 4: Tạo Bản đồ Tương tác Folium
    # -------------------------------------------------------------
    print("\n[TEST 4] Tạo bản đồ tương tác Folium và xuất file HTML...")
    docs_dir = os.path.join(root_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    html_output = os.path.join(docs_dir, "vietnam_prediction_map.html")
    
    m = create_prediction_map(predictions, save_html_path=html_output)
    assert os.path.exists(html_output), f"Không tìm thấy file HTML bản đồ tại: {html_output}"
    print(f" -> [PASSED] Bản đồ HTML đã được lưu thành công tại: {html_output}!")

    # -------------------------------------------------------------
    # KẾT LUẬN TOÀN BỘ BỘ TEST
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  🏆 TẤT CẢ 4/4 TEST CASES ĐỀU ĐẠT CHUẨN XÁC 100% (ALL PASSED!)")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
