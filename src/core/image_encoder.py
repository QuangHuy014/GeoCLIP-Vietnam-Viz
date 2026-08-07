# ==============================================================================
# MODULE: Image Encoder (Tech Lead: Dev 1)
# RESPONSIBILITY: Load CLIP Vision Backbone & extract image embeddings (512-d)
# ==============================================================================

import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # TODO (Tech Lead): Initialize CLIP Model and MLP projection head (768 -> 512)
        pass

    def preprocess_image(self, image):
        # TODO (Tech Lead): Preprocess PIL image using CLIP processor
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (Tech Lead): Extract CLIP image features & project to 512-d
        pass
