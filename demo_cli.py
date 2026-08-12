# ==============================================================================
# ENTRYPOINT: CLI Demo Runner (Tech Lead)
# RESPONSIBILITY: Simple CLI script for testing prediction flow in terminal
# ==============================================================================

import os
import sys
from PIL import Image

# Thêm thư mục gốc vào sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Đảm bảo in được tiếng Việt có dấu trên mọi Windows CMD
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.core.pipeline import GeoCLIPService

def main():
    print("=" * 65)
    print("      🌍 GEOCLIP VIETNAM VISUALIZATION - CLI PREDICTION DEMO")
    print("=" * 65)

    # 1. Nhận đường dẫn ảnh từ tham số dòng lệnh hoặc dùng ảnh mẫu
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Ưu tiên tìm các ảnh có sẵn trong thư mục data/
        candidates = [
            os.path.join(root_dir, "data", "sample_images", "Kauai.png"),
            os.path.join(root_dir, "data", "images.jpg"),
            os.path.join(root_dir, "data", "sample_images", "images.jpg")
        ]
        image_path = None
        for p in candidates:
            if os.path.exists(p):
                image_path = p
                break

    if not image_path or not os.path.exists(image_path):
        print(f"[!] Không tìm thấy file ảnh tại: {image_path}")
        print("[!] Cách sử dụng: python demo_cli.py <duong_dan_anh.jpg>")
        return

    print(f"\n[+] Đang nạp mô hình và quét ảnh: {os.path.basename(image_path)}...")
    service = GeoCLIPService(root_dir=root_dir)

    print(f"\n[+] Đang tiến hành dự đoán Top-5 vị trí địa lý...")
    results = service.predict(image_path, top_k=5)

    print("\n" + "=" * 65)
    print("               📍 KẾT QUẢ DỰ ĐOÁN TOP-5 VỊ TRÍ")
    print("=" * 65)
    for res in results:
        print(f"Hạng #{res['rank']}: Vĩ độ = {res['lat']:10.6f}, Kinh độ = {res['lon']:10.6f} | Xác suất: {res['prob_percent']:6.2f}%")
        print(f"        🔗 Google Maps: {res['gmaps_url']}")
    print("=" * 65)
    print("[OK] Dự đoán hoàn tất thành công!\n")

if __name__ == "__main__":
    main()
