import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image

root_dir = os.path.abspath(".")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.core.pipeline import GeoCLIPService
from src.core.matcher import CosineMatcher

service = GeoCLIPService()
test_img = "data/images.jpg"
image = Image.open(test_img).convert("RGB")
pixel_values = service.image_encoder.preprocess_image(image)

with torch.no_grad():
    image_features = service.image_encoder(pixel_values)
    logits = CosineMatcher.match(image_features, service.location_features, service.logit_scale)
    probs = logits.softmax(dim=-1)[0]

df = service.metadata_df

# 1. Tìm các điểm trong bán kính Landmark 81 (Lat: 10.795, Lon: 106.721)
l81_mask = (df['LAT'] >= 10.78) & (df['LAT'] <= 10.81) & (df['LON'] >= 106.70) & (df['LON'] <= 106.74)
l81_indices = np.where(l81_mask)[0]

# 2. Tìm toàn bộ điểm tại TP.HCM (Lat: 10.65 - 10.90, Lon: 106.55 - 106.85)
hcm_mask = (df['LAT'] >= 10.65) & (df['LAT'] <= 10.90) & (df['LON'] >= 106.55) & (df['LON'] <= 106.85)
hcm_indices = np.where(hcm_mask)[0]

# Sắp xếp thứ hạng toàn quốc
sorted_indices = probs.argsort(descending=True).cpu().numpy()
ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}

print(f"\n============================================================")
print(f"       PHAN TICH CHUYEN SAU DU DOAN CHO LANDMARK 81")
print(f"============================================================")
print(f"Tong so diem POI toan Viet Nam: {len(df):,}")
print(f"Tong so diem POI tai TP.HCM: {len(hcm_indices):,}")
print(f"So diem POI ngay sat Landmark 81 (1-2km): {len(l81_indices)}")

if len(l81_indices) > 0:
    l81_probs = probs[l81_indices].cpu().numpy()
    best_l81_local = l81_probs.argmax()
    best_l81_idx = l81_indices[best_l81_local]
    
    print(f"\n[Diem Landmark 81 dat diem cao nhat]:")
    print(f" - Ten dia danh: {df.iloc[best_l81_idx]['NAME']}")
    print(f" - Loai hinh: {df.iloc[best_l81_idx]['AMENITY']}")
    print(f" - Toa do: Lat = {df.iloc[best_l81_idx]['LAT']:.6f}, Lon = {df.iloc[best_l81_idx]['LON']:.6f}")
    print(f" - Xac suat: {probs[best_l81_idx].item() * 100:.4f}%")
    print(f" - Thu hang (Rank): #{ranks[best_l81_idx]:,} / {len(df):,}")

# Top 1 TP.HCM
if len(hcm_indices) > 0:
    hcm_probs = probs[hcm_indices].cpu().numpy()
    best_hcm_local = hcm_probs.argmax()
    best_hcm_idx = hcm_indices[best_hcm_local]
    print(f"\n[Diem cao nhat tai khu vuc TP.HCM]:")
    print(f" - Ten dia danh: {df.iloc[best_hcm_idx]['NAME']}")
    print(f" - Toa do: Lat = {df.iloc[best_hcm_idx]['LAT']:.6f}, Lon = {df.iloc[best_hcm_idx]['LON']:.6f}")
    print(f" - Xac suat: {probs[best_hcm_idx].item() * 100:.4f}%")
    print(f" - Thu hang (Rank): #{ranks[best_hcm_idx]:,} / {len(df):,}")

# Top 3 toan quoc
print(f"\n[TOP 3 DIEM DUOC MO HINH CHON TRONG THUC TE]:")
for r in range(3):
    top_idx = sorted_indices[r]
    row = df.iloc[top_idx]
    print(f"Rank #{r+1}: {row['NAME']} ({row['AMENITY']}) | Lat = {row['LAT']:.4f}, Lon = {row['LON']:.4f} | Prob = {probs[top_idx].item()*100:.4f}%")
