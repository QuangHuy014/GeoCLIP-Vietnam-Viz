# ==============================================================================
# MODULE: Random Fourier Features RFF (GIS Analyst: Dev 2)
# RESPONSIBILITY: Gaussian Encoding layer for multi-scale frequency mapping
# ==============================================================================

import math
from typing import List, Union
import torch
import torch.nn as nn


class GaussianEncoding(nn.Module):
    """
    Gaussian Fourier Feature Encoding Layer.
    Maps 2D spatial coordinates (x, y) into high-dimensional frequency space using
    random Gaussian projection matrices: v -> [cos(2*pi*v*B), sin(2*pi*v*B)].
    """
    #sigma mức tần số, input đầu ra có 2 (x, y),
    def __init__(self, sigma: float, input_size: int = 2, encoded_size: int = 256, seed: int = 42):
        super().__init__()
        self.sigma = float(sigma)
        self.input_size = input_size
        self.encoded_size = encoded_size

        # Generate Gaussian random matrix B ~ N(0, sigma^2)
        generator = torch.Generator()
        generator.manual_seed(seed)
        B = torch.randn((input_size, encoded_size), generator=generator) * self.sigma

        #xem B là một phần của model, nhưng nó không phải thứ cần học.
        self.register_buffer("B", B)

    #hàm được gọi khi bạn viết: rff(xy)
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Gaussian Encoding.
        Args:
            v (torch.Tensor): Tensor of shape (N, 2) or (2,) containing (x, y) coordinates.
        Returns:
            torch.Tensor: Encoded Fourier features of shape (N, 2 * encoded_size).
        """

        #Nếu bạn truyền: rff([[0.5, 0.5]]) thì Python list được chuyển thành: Tensor
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32)

        #Xử lý tọa độ 1 chiều kết quả: [0.5, 0.5]
        is_1d = v.dim() == 1
        if is_1d:
            v = v.unsqueeze(0)

        # v: (N, 2), B: (2, encoded_size) -> v_proj: (N, encoded_size) 2πvB
        v_proj = 2.0 * math.pi * torch.matmul(v, self.B)
        
        # Output [cos(v_proj), sin(v_proj)] along last dimension -> (N, 2 * encoded_size)
        out = torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)

        if is_1d:
            out = out.squeeze(0)
        return out

#Nó không chỉ dùng một GaussianEncoding, mà dùng 3 cái sigma = 1, sigma = 16, sigma = 256
class MultiScaleGaussianEncoding(nn.Module):
    """
    Multi-scale Gaussian Fourier Features module combining multiple sigma frequencies:
    sigma = [2^0, 2^4, 2^8] = [1, 16, 256] (Continent -> Country -> Region).
    """

    def __init__(
        self,
        sigmas: List[float] = [1.0, 16.0, 256.0],
        input_size: int = 2,
        encoded_size_per_scale: int = 256,
        #Mục đích là mỗi encoder có ma trận random B khác nhau.
        seed: int = 42,
    ):
        super().__init__()
        self.sigmas = [float(s) for s in sigmas]
        self.encoders = nn.ModuleList(
            [
                GaussianEncoding(
                    sigma=s,
                    input_size=input_size,
                    encoded_size=encoded_size_per_scale,
                    seed=seed + i,
                )
                for i, s in enumerate(self.sigmas)
            ]
        )
        self.out_dim = len(self.sigmas) * (2 * encoded_size_per_scale)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32)

        is_1d = v.dim() == 1
        if is_1d:
            v = v.unsqueeze(0)

        features = [encoder(v) for encoder in self.encoders]
        out = torch.cat(features, dim=-1)

        if is_1d:
            out = out.squeeze(0)
        return out
