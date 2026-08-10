# ==============================================================================
# MODULE: Image Encoder (Tech Lead: Dev 1)
# RESPONSIBILITY: Load CLIP Vision Backbone & extract image embeddings (512-d)
# ==============================================================================

import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, AutoProcessor
import warnings

# Tắt các cảnh báo không ảnh hưởng của HuggingFace Hub
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')


class StandaloneImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "openai/clip-vit-large-patch14",
                 mlp_weights_path: str = None,
                 device: str = None):
        super(StandaloneImageEncoder, self).__init__()

        # 1. Tự động xác định thiết bị (cuda nếu có, không thì cpu)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 2. Tải Backbone CLIP và bộ xử lý ảnh từ HuggingFace
        self.CLIP = CLIPModel.from_pretrained(model_name).to(self.device)
        self.image_processor = AutoProcessor.from_pretrained(model_name)

        # 3. Định nghĩa mạng MLP chiếu từ 768 chiều -> 512 chiều
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512)
        ).to(self.device)

        # 4. Tự động nạp trọng số pretrained nếu có đường dẫn
        if mlp_weights_path and os.path.exists(mlp_weights_path):
            state_dict = torch.load(mlp_weights_path, map_location=self.device)
            self.mlp.load_state_dict(state_dict)
            print(f"[ImageEncoder] Loaded MLP weights from: {mlp_weights_path}")

        # 5. Đóng băng (Freeze) CLIP để không tính gradient (tiết kiệm RAM & chạy nhanh)
        for param in self.CLIP.parameters():
            param.requires_grad = False
        self.eval()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Nhận ảnh PIL -> Trả về Tensor pixel chuẩn hóa kích thước (1, 3, 224, 224)
        """
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Tensor ảnh (1, 3, 224, 224) -> Vector đặc trưng (1, 512)
        """
        # 1. Trích xuất đặc trưng thị giác từ CLIP (768 chiều)
        x = self.CLIP.get_image_features(pixel_values=pixel_values)

        # 2. Xử lý tương thích nếu HuggingFace trả về class BaseModelOutputWithPooling
        if not isinstance(x, torch.Tensor):
            x = getattr(x, "image_embeds", getattr(x, "pooler_output", x[0]))

        # 3. Đưa qua mạng MLP chiếu về 512 chiều
        x = self.mlp(x)
        return x


# ==============================================================================
# HÀM TIỆN ÍCH CẤP CAO (Định nghĩa ngoài class)
# ==============================================================================
def extract_image_features(image_input, encoder: StandaloneImageEncoder = None) -> torch.Tensor:
    """
    Hàm tiện ích cấp cao: Nhận đường dẫn file ảnh (str) hoặc ảnh (PIL Image)
    -> Trả về Vector Tensor (1, 512)
    """
    if encoder is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.abspath(
            os.path.join(current_dir, "..", "..", "weights", "image_encoder_mlp_weights.pth"))
        encoder = StandaloneImageEncoder(mlp_weights_path=weights_path)

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    pixel_values = encoder.preprocess_image(image)
    with torch.no_grad():
        features = encoder(pixel_values)
    return features


# ==============================================================================
# UNIT TEST
# ==============================================================================
if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("      [Tech Lead Unit Test] Testing StandaloneImageEncoder")
    print("=" * 60)

    encoder = StandaloneImageEncoder()
    dummy_array = np.uint8(np.random.rand(224, 224, 3) * 255)
    dummy_img = Image.fromarray(dummy_array)

    feat = extract_image_features(dummy_img, encoder=encoder)

    print(f"[OK] Extracted Feature Tensor Shape: {feat.shape}")
    print(f"[OK] Device used: {feat.device}")
    assert feat.shape == (1, 512), "Kích thước đầu ra bắt buộc phải là (1, 512)!"
    print("[Tech Lead Unit Test] Task 1.1 StandaloneImageEncoder PASSED!")
