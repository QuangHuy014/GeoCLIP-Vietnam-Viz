# ==============================================================================
# MODULE: GeoCLIP Main Pipeline Service (Tech Lead: Dev 1)
# RESPONSIBILITY: Main API Integration Service for predicting GPS from image
# ==============================================================================

import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict

# Thêm thư mục gốc của project vào sys.path để import được gói 'src'
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Đảm bảo in được tiếng Việt có dấu trên mọi Windows CMD
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.core.image_encoder import StandaloneImageEncoder
from src.core.matcher import CosineMatcher
from src.core.location_encoder import LocationEncoder

class GeoCLIPService:
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.weights_dir = os.path.join(root_dir, "weights")
        self.data_dir = os.path.join(root_dir, "data")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Nạp các mô hình AI
        self._load_model()
        
        # 2. Nạp dữ liệu tọa độ 100K toàn cầu và tiền tính toán Location Embeddings (Caching)
        self._load_gps_gallery()

    def _load_model(self):
        """Khởi tạo ImageEncoder, LocationEncoder và Logit Scale"""
        print(f"[GeoCLIPService] Initializing models on device: {self.device}...")

        # Nạp ImageEncoder (Task 1.1)
        img_mlp_weights = os.path.join(self.weights_dir, "image_encoder_mlp_weights.pth")
        self.image_encoder = StandaloneImageEncoder(mlp_weights_path=img_mlp_weights, device=self.device)

        # Nạp LocationEncoder (Trọng số không gian)
        loc_weights = os.path.join(self.weights_dir, "location_encoder_weights.pth")
        self.location_encoder = LocationEncoder(weights_path=loc_weights).to(self.device)

        # Nạp tham số nhiệt độ logit_scale
        logit_weights = os.path.join(self.weights_dir, "logit_scale_weights.pth")
        if os.path.exists(logit_weights):
            self.logit_scale = torch.load(logit_weights, map_location=self.device)
        else:
            self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
            
        self.image_encoder.eval()
        self.location_encoder.eval()

    def _load_gps_gallery(self):
        """
        Nạp tập dữ liệu 100K tọa độ phân bố toàn cầu (coordinates_100K.csv)
        """
        global_csv = os.path.join(self.data_dir, "coordinates_100K.csv")
        if not os.path.exists(global_csv):
            raise FileNotFoundError(f"Không tìm thấy file coordinates_100K.csv tại: {global_csv}!")

        print(f"[GeoCLIPService] Reading Global GPS Gallery: {os.path.basename(global_csv)}...")
        df_global = pd.read_csv(global_csv)[['LAT', 'LON']].drop_duplicates().reset_index(drop=True)
        self.gps_gallery = torch.tensor(df_global[['LAT', 'LON']].values, dtype=torch.float32)
        print(f"[GeoCLIPService] Loaded {len(self.gps_gallery):,} Global GPS coordinates into Gallery!")

        # Tiền tính toán Location Embeddings vào RAM (Pre-encoding)
        print(f"[GeoCLIPService] Pre-encoding Location Embeddings...")
        batch_size = 16384
        loc_features_list = []
        with torch.no_grad():
            for i in range(0, len(self.gps_gallery), batch_size):
                batch_gps = self.gps_gallery[i:i+batch_size].to(self.device)
                feat = self.location_encoder(batch_gps)
                loc_features_list.append(feat.cpu())
        self.location_features = torch.cat(loc_features_list, dim=0).to(self.device)
        print(f"[GeoCLIPService] Global Location Embeddings Ready! Matrix Shape: {self.location_features.shape}")

    def predict(self, image_input, top_k: int = 5) -> List[Dict]:
        """
        Dự đoán Top-K tọa độ GPS toàn cầu cho 1 bức ảnh đầu vào (Đường dẫn str hoặc PIL Image)
        Thời gian suy luận siêu tốc: ~0.02s
        """
        # 1. Đọc và tiền xử lý ảnh
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        pixel_values = self.image_encoder.preprocess_image(image)

        # 2. Trích xuất đặc trưng ảnh & So khớp Cosine siêu tốc với ma trận đã tiền tính toán
        with torch.no_grad():
            image_features = self.image_encoder(pixel_values)

            # Gọi hàm match và get_top_k từ CosineMatcher (Task 1.2)
            logits = CosineMatcher.match(image_features, self.location_features, self.logit_scale)
            indices, probs = CosineMatcher.get_top_k(logits, top_k=top_k)

        # 3. Đóng gói kết quả sạch để bàn giao cho UI Team
        results = []
        for i in range(top_k):
            idx = indices[i].item()
            lat = float(self.gps_gallery[idx][0].item())
            lon = float(self.gps_gallery[idx][1].item())
            prob = float(probs[i].item() * 100)

            results.append({
                "rank": i + 1,
                "lat": lat,
                "lon": lon,
                "prob_percent": round(prob, 2),
                "gmaps_url": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
            })

        return results


# ==============================================================================
# UNIT TEST (Kiểm thử dịch vụ với ảnh thực tế)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("      [Tech Lead Unit Test] Testing GeoCLIPService (Global 100K)")
    print("=" * 60)

    # 1. Khởi tạo Service
    service = GeoCLIPService()

    # 2. Tìm ảnh mẫu để test (Ưu tiên ảnh mẫu Kauai.png từ tác giả GeoCLIP)
    candidate_paths = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        os.path.join(root_dir, "data", "sample_images", "Kauai.png"),
        os.path.join(root_dir, "data", "images.jpg"),
        os.path.join(root_dir, "data", "sample_images", "images.jpg")
    ]
    test_img = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            test_img = p
            break

    if test_img is None:
        dummy_array = np.uint8(np.random.rand(224, 224, 3) * 255)
        test_img = Image.fromarray(dummy_array)

    # 3. Chạy hàm predict
    print(f"\n[Test] Predicting location for: {test_img}...")
    predictions = service.predict(test_img, top_k=5)

    print("\n" + "-" * 60)
    print("              TOP 5 GLOBAL PREDICTION RESULTS")
    print("-" * 60)
    for p in predictions:
        print(f"Rank #{p['rank']}: Lat = {p['lat']:10.6f}, Lon = {p['lon']:10.6f} | Probability: {p['prob_percent']:6.2f}%")
        print(f"        Google Maps: {p['gmaps_url']}")

    assert len(predictions) == 5, "Phải trả về đúng 5 kết quả!"
    print("\n[Tech Lead Unit Test] Task 1.3 GeoCLIPService (Global 100K) PASSED!")
