# ==============================================================================
# MODULE: Random Fourier Features RFF (GIS Analyst: Dev 2)
# RESPONSIBILITY: Gaussian Encoding layer for multi-scale frequency mapping
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np

class GaussianEncoding(nn.Module):
    """
    Lớp mã hóa Fourier ngẫu nhiên (RFF) dùng phân phối Gaussian.
    """
    def __init__(self, sigma: float, input_size: int = 2, encoded_size: int = 256):
        super().__init__()
        # Khởi tạo ma trận ngẫu nhiên B ~ N(0, sigma^2)
        b = torch.randn((encoded_size, input_size)) * sigma
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # gamma(v) = [cos(2*pi*B*v), sin(2*pi*B*v)]
        vp = 2 * np.pi * torch.matmul(v, self.b.t())
        return torch.cat([torch.cos(vp), torch.sin(vp)], dim=-1)
