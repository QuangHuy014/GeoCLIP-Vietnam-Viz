# ==============================================================================
# MODULE: Cosine Matcher (Tech Lead: Dev 1)
# RESPONSIBILITY: Compute Cosine Similarity & Softmax Top-K probabilities
# ==============================================================================

import torch
import torch.nn.functional as F
from typing import Tuple

class CosineMatcher:
    @staticmethod
    def match(image_features: torch.Tensor,
              location_features: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
        """
        Tính điểm tương đồng Cosine Similarity giữa Vector Ảnh và Ma trận Vector Vị trí
        Args:
            image_features: Tensor (1, 512)
            location_features: Tensor (M, 512)
            logit_scale: Tensor chứa tham số nhiệt độ tau
        Returns:
            logits: Tensor điểm số (1, M)
        """
        # 1. Chuẩn hóa L2 vector ảnh và vector vị trí về độ dài = 1
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # 2. Tính hệ số nhiệt độ scale = e^(logit_scale)
        scale = logit_scale.exp()

        # 3. Nhân ma trận: (1, 512) @ (512, M) -> Trả về Logits (1, M)
        logits = scale * (image_features @ location_features.t())
        return logits

    @staticmethod
    def get_top_k(logits: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Đưa qua Softmax và lọc ra Top-K chỉ số có xác suất cao nhất
        Args:
            logits: Tensor (1, M)
            top_k: Số lượng vị trí cần lấy (mặc định = 5)
        Returns:
            indices: Mảng chỉ số của Top-K vị trí (K,)
            probabilities: Mảng giá trị % xác suất tương ứng (K,)
        """
        # 1. Chuyển đổi điểm Logits thành phân bố xác suất Softmax
        probs = logits.softmax(dim=-1)

        # 2. Trích xuất Top K giá trị lớn nhất
        top_pred = torch.topk(probs, top_k, dim=1)

        indices = top_pred.indices[0]
        probabilities = top_pred.values[0]
        return indices, probabilities


# ==============================================================================
# UNIT TEST (Kiểm thử độc lập module này)
# ==============================================================================
if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("      [Tech Lead Unit Test] Testing CosineMatcher")
    print("=" * 60)

    # 1. Giả lập 1 vector ảnh (1, 512)
    dummy_img = torch.randn(1, 512)

    # 2. Giả lập 100,000 vector vị trí trên Trái Đất (100000, 512)
    dummy_locs = torch.randn(100000, 512)

    # 3. Giả lập tham số nhiệt độ logit_scale
    dummy_logit_scale = torch.tensor(np.log(1 / 0.07))

    # 4. Kiểm tra hàm match
    logits = CosineMatcher.match(dummy_img, dummy_locs, dummy_logit_scale)
    print(f"[OK] Logits Matrix Shape: {logits.shape}")
    assert logits.shape == (1, 100000), "Shape Logits bắt buộc phải là (1, 100000)!"

    # 5. Kiểm tra hàm get_top_k
    indices, probs = CosineMatcher.get_top_k(logits, top_k=5)
    print(f"[OK] Top-5 Indices: {indices.tolist()}")
    print(f"[OK] Top-5 Probabilities: {[round(p.item() * 100, 2) for p in probs]}%")
    assert len(indices) == 5 and len(probs) == 5, "Phải trả về đúng 5 phần tử!"

    print("[Tech Lead Unit Test] Task 1.2 CosineMatcher PASSED!")
