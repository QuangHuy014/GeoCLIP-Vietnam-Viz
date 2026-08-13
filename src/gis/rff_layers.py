# ==============================================================================
# MODULE: Random Fourier Features RFF (Tần số sóng ngẫu nhiên Fourier)
# TRÁCH NHIỆM: Mã hóa không gian đa tần số Gaussian cho tọa độ địa lý
# ==============================================================================

import sys
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np


class GaussianEncoding(nn.Module):
    """
    Lớp mã hóa Fourier ngẫu nhiên (RFF) sử dụng phân phối Gaussian:
    Chuyển đổi tọa độ 2D (x, y) -> [cos(2*pi*v*B), sin(2*pi*v*B)].
    """

    def __init__(
        self,
        sigma: float,
        input_size: int = 2,
        encoded_size: int = 256,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.sigma = float(sigma)
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            b = torch.randn((encoded_size, input_size), generator=generator) * float(sigma)
        else:
            b = torch.randn((encoded_size, input_size)) * float(sigma)

        # Lưu ma trận tần số b làm tham số không thay đổi gradient
        self.b = nn.Parameter(b, requires_grad=False)

    @property
    def B(self):
        """Thuộc tính truy cập ma trận B cho tính tương thích ngược."""
        return self.b.t()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32)

        is_1d = v.dim() == 1
        if is_1d:
            v = v.unsqueeze(0)

        # gamma(v) = [cos(2*pi*B*v), sin(2*pi*B*v)]
        vp = 2.0 * np.pi * torch.matmul(v, self.b.t())
        out = torch.cat([torch.cos(vp), torch.sin(vp)], dim=-1)

        if is_1d:
            out = out.squeeze(0)
        return out


class MultiScaleGaussianEncoding(nn.Module):
    """
    Mô-đun mã hóa RFF đa tỷ lệ tần số combining các mức sigma:
    sigma = [2^0, 2^4, 2^8] = [1, 16, 256] (Châu lục -> Quốc gia -> Khu vực địa danh).
    """

    def __init__(
        self,
        sigmas: List[float] = [1.0, 16.0, 256.0],
        input_size: int = 2,
        encoded_size_per_scale: int = 256,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.sigmas = [float(s) for s in sigmas]
        self.encoders = nn.ModuleList(
            [
                GaussianEncoding(
                    sigma=s,
                    input_size=input_size,
                    encoded_size=encoded_size_per_scale,
                    seed=(seed + i) if seed is not None else None,
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


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("======================================================================")
    print("      CHƯƠNG TRÌNH CHẠY THỬ MÔ-ĐUN MÃ HÓA TẦN SỐ RFF (GAUSSIAN)        ")
    print("======================================================================\n")

    # 1. Khởi tạo RFF với 3 mức sigma: 1.0 (Châu lục), 16.0 (Quốc gia), 256.0 (Địa danh)
    rff = MultiScaleGaussianEncoding(sigmas=[1.0, 16.0, 256.0])

    # 2. Truyền tọa độ 2D giả lập
    xy_dummy = torch.tensor([[0.45, 0.82]])
    rff_output = rff(xy_dummy)

    print(f"1. Tọa độ đầu vào  : {xy_dummy.numpy()} (Shape: {xy_dummy.shape})")
    print(f"2. Đầu ra sau RFF  : Shape {rff_output.shape} (1536 chiều từ 3 dải tần số)")
    print(f"3. 5 Giá trị đầu tiên: {rff_output[0, :5].numpy()}")

    # 3. Kiểm tra tính ổn định (cùng input -> kết quả trùng 100%)
    rff_output2 = rff(xy_dummy)
    is_same = torch.equal(rff_output, rff_output2)
    print(f"\n✅ Kiểm tra tính ổn định: {is_same} (Ma trận tần số không bị random lại)")
    print("======================================================================")
