# ==============================================================================
# MODULE: GeoCLIP Main Pipeline Service (Tech Lead & Agent Team)
# RESPONSIBILITY: Main API Integration Service for predicting GPS in Vietnam & Globally
# ==============================================================================

import os
import sys
import json
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
    def __init__(self, root_dir: str = None, scope: str = "vietnam"):
        """
        Khởi tạo GeoCLIP Service.
        Args:
            root_dir: Đường dẫn thư mục gốc của dự án.
            scope: 'vietnam' (Chuyên biệt Việt Nam) hoặc 'global' (Toàn cầu 100K).
        """
        if root_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.root_dir = root_dir
        self.weights_dir = os.path.join(root_dir, "weights")
        self.data_dir = os.path.join(root_dir, "data")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scope = scope.lower()
        
        # 1. Nạp các mô hình AI Backbone
        self._load_model()
        
        # 2. Nạp dữ liệu tọa độ và tiền tính toán Location Embeddings (RAM Caching)
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
        Nạp tập tọa độ tương ứng theo chế độ: 'vietnam_iconic', 'vietnam_all', 'vietnam' hoặc 'global'
        """
        if self.scope in ["vietnam_iconic", "iconic"]:
            vn_csv = os.path.join(self.data_dir, "vietnam_landmarks_iconic.csv")
            if not os.path.exists(vn_csv):
                vn_csv = os.path.join(self.data_dir, "vietnam_landmarks.csv")

            print(f"[GeoCLIPService] Loading Vietnam Iconic Landmarks: {os.path.basename(vn_csv)}...")
            self.metadata_df = pd.read_csv(vn_csv).reset_index(drop=True)
            self.gps_gallery = torch.tensor(self.metadata_df[['LAT', 'LON']].values, dtype=torch.float32)
            print(f"[GeoCLIPService] Loaded {len(self.gps_gallery):,} Vietnam Iconic Landmarks!")
        elif self.scope in ["vietnam", "vietnam_all", "all"]:
            vn_csv = os.path.join(self.data_dir, "vietnam_landmarks.csv")
            if not os.path.exists(vn_csv):
                raise FileNotFoundError(f"Không tìm thấy file: {vn_csv}")

            print(f"[GeoCLIPService] Loading Vietnam Full POI Dataset: {os.path.basename(vn_csv)}...")
            self.metadata_df = pd.read_csv(vn_csv).reset_index(drop=True)
            self.gps_gallery = torch.tensor(self.metadata_df[['LAT', 'LON']].values, dtype=torch.float32)
            print(f"[GeoCLIPService] Loaded {len(self.gps_gallery):,} Vietnam POI coordinates into Gallery!")
        else:
            global_csv = os.path.join(self.data_dir, "coordinates_100K.csv")
            if not os.path.exists(global_csv):
                raise FileNotFoundError(f"Không tìm thấy file: {global_csv}")

            print(f"[GeoCLIPService] Loading Global 100K GPS Gallery: {os.path.basename(global_csv)}...")
            df_global = pd.read_csv(global_csv)[['LAT', 'LON']].drop_duplicates().reset_index(drop=True)
            self.metadata_df = df_global
            self.gps_gallery = torch.tensor(df_global[['LAT', 'LON']].values, dtype=torch.float32)
            print(f"[GeoCLIPService] Loaded {len(self.gps_gallery):,} Global GPS coordinates!")

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
        print(f"[GeoCLIPService] Location Embeddings Ready! Matrix Shape: {self.location_features.shape}")

    def predict(self, image_input, top_k: int = 5) -> List[Dict]:
        """
        Dự đoán Top-K tọa độ GPS cho 1 bức ảnh đầu vào (Đường dẫn str hoặc PIL Image)
        Thời gian suy luận: ~0.005s
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
            top_k_val = min(top_k, len(self.gps_gallery))
            indices, probs = CosineMatcher.get_top_k(logits, top_k=top_k_val)

        # 3. Đóng gói kết quả sạch kèm Rich Metadata
        results = []
        for i in range(top_k_val):
            idx = indices[i].item()
            row = self.metadata_df.iloc[idx]
            lat = float(row['LAT'])
            lon = float(row['LON'])
            prob = float(probs[i].item() * 100)

            name = str(row.get('NAME', f'GPS Point #{idx}'))
            category = str(row.get('CATEGORY', 'Địa danh'))
            province = str(row.get('PROVINCE', 'Việt Nam'))
            description = str(row.get('DESCRIPTION', ''))

            results.append({
                "rank": i + 1,
                "name": name,
                "category": category,
                "province": province,
                "description": description,
                "lat": lat,
                "lon": lon,
                "prob_percent": round(prob, 2),
                "gmaps_url": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
            })

        return results


# ==============================================================================
# UNIT TEST (Kiểm thử dịch vụ với ảnh Landmark 81)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("      [Tech Lead Unit Test] Testing GeoCLIPService (Vietnam Mode)")
    print("=" * 65)

    # 1. Khởi tạo Service chế độ Việt Nam
    service = GeoCLIPService(scope="vietnam")

    # 2. Tìm ảnh Landmark 81
    candidate_paths = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        os.path.join(root_dir, "data", "images.jpg"),
        os.path.join(root_dir, "data", "sample_images", "images.jpg")
    ]
    test_img = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            test_img = p
            break

    if test_img is None:
        print("[!] Không tìm thấy ảnh images.jpg!")
        sys.exit(1)

    # 3. Chạy hàm predict
    print(f"\n[Test] Predicting location for: {test_img}...")
    predictions = service.predict(test_img, top_k=5)

    print("\n" + "=" * 65)
    print("              📍 TOP 5 VIETNAM PREDICTION RESULTS")
    print("=" * 65)
    for p in predictions:
        print(f"Hạng #{p['rank']}: {p['name']} ({p['province']}) - {p['category']}")
        print(f"        Vĩ độ = {p['lat']:10.6f}, Kinh độ = {p['lon']:10.6f} | Xác suất: {p['prob_percent']:6.2f}%")
        print(f"        Google Maps: {p['gmaps_url']}")

    assert len(predictions) >= 1, "Phải có kết quả trả về!"
    assert predictions[0]['name'] == 'Landmark 81', f"Top 1 phải là Landmark 81 nhưng lại là {predictions[0]['name']}!"
    print("\n[Tech Lead Unit Test] Task 1.3 GeoCLIPService (Vietnam Mode) PASSED!")
