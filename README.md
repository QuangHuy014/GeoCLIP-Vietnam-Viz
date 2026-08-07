# 🌍 GeoCLIP Vietnam Visualization (`GeoCLIP-Vietnam-Viz`)
> **Đồ án môn Phân tích và Trực quan hóa Dữ liệu - Trường ĐH CNTT (UIT)**  
> Dự đoán vị trí địa lý của hình ảnh sử dụng Contrastive Vision-Location Matching (GeoCLIP), làm phong phú dữ liệu địa danh Việt Nam và trực quan hóa dữ liệu không gian.

---

## 📌 1. Giới thiệu Đồ án
Dự án tập trung vào bài toán **Geo-localization (Dự đoán vị trí địa lý từ hình ảnh)** dựa trên kiến trúc **GeoCLIP**:
- **Khối Thị giác (Vision)**: Trích xuất đặc trưng hình ảnh sử dụng mô hình thị giác CLIP ViT-L/14 của OpenAI.
- **Khối Địa lý (GIS)**: Chiếu tọa độ cầu WGS84 về mặt phẳng 2D (**Equal Earth Projection**) và mã hóa tần số đa tỷ lệ bằng **Random Fourier Features (RFF)**.
- **Đồ án môn học**: Bổ sung tập dữ liệu các địa danh nổi tiếng ở Việt Nam, khắc phục hiện tượng Data Bias và xây dựng Dashboard trực quan hóa không gian tương tác.

---

## 📂 2. Cấu trúc Thư mục Dự án (Project Framework)

```text
GeoCLIP-Vietnam-Viz/
├── 📄 .gitignore                 # File cấu hình Git ignore
├── 📄 requirements.txt           # Danh sách các thư viện phụ thuộc
├── 📘 README.md                  # Hướng dẫn dự án
├── 📋 PLAN_TEAM.md               # Kế hoạch & Phân công Task 2 tuần cho nhóm
├── 🖥️ app.py                     # [UI Team] Web Dashboard Streamlit
├── ⚡ demo_cli.py                 # [Tech Lead] Script thử nghiệm Terminal
├── 📂 weights/                   # Lưu trữ các tệp trọng số pretrained (.pth)
│   ├── image_encoder_mlp_weights.pth
│   ├── location_encoder_weights.pth
│   └── logit_scale_weights.pth
├── 📂 data/                      # Dữ liệu Tọa độ & Ảnh mẫu
│   ├── coordinates_100K.csv     # 100,000 tọa độ mẫu toàn cầu
│   └── vietnam_landmarks.csv    # Tập dữ liệu tọa độ địa danh Việt Nam
├── 📂 notebooks/                 # [GIS Team] Jupyter Notebooks cho EDA & Phân tích sai số
│   ├── 1_geospatial_eda.ipynb
│   └── 2_model_evaluation.ipynb
└── 📂 src/                       # Mã nguồn mô-đun hóa chính
    ├── 📂 core/                  # [Tech Lead - Dev 1] Core AI Engine
    │   ├── image_encoder.py     # Stub: CLIP Vision Backbone
    │   ├── location_encoder.py  # Stub: Multi-scale Capsule Location Encoder
    │   ├── matcher.py           # Stub: Cosine Similarity & Softmax Matcher
    │   └── pipeline.py          # Stub: Class GeoCLIPService API chung
    ├── 📂 gis/                   # [GIS Engineer - Dev 2] GIS & Data Metrics
    │   ├── projections.py       # Stub: Phép chiếu Equal Earth
    │   ├── rff_layers.py        # Stub: Random Fourier Features (RFF)
    │   └── distance_metrics.py  # Stub: Phân tích khoảng cách sai số (km)
    └── 📂 viz/                   # [Frontend Team - 3 thành viên] Bản đồ & Chart
        └── map_builder.py       # Stub: Dựng bản đồ tương tác Leaflet/Folium
```

---

## 👥 3. Phân chia Trách nhiệm Thành viên (Team Roles)

| Vai trò | Thành viên | Phụ trách chính | Thư mục/File đảm nhận |
| :--- | :--- | :--- | :--- |
| **Tech Lead** | Bạn | Kiến trúc AI Core, Integration, Service API | `src/core/`, `demo_cli.py` |
| **GIS Engineer** | Thành viên 2 | Phép chiếu Equal Earth, RFF, Dữ liệu VN & Metric sai số | `src/gis/`, `data/`, `notebooks/` |
| **Frontend UI** | 3 Thành viên | Web Dashboard Streamlit, Bản đồ tương tác & Biểu đồ | `src/viz/`, `app.py` |

---

## 🛠️ 4. Hướng dẫn Khởi chạy Dự án (Getting Started)

### Bước 1: Tạo môi trường ảo & Cài đặt Thư viện
```powershell
# Tạo và kích hoạt virtual environment
python -m venv venv
.\venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Chạy thử nghiệm CLI (Terminal)
```powershell
python demo_cli.py
```

### Bước 3: Khởi chạy Web Dashboard Streamlit
```powershell
streamlit run app.py
```
*(Giao diện Web sẽ mở tại địa chỉ `http://localhost:8501`)*

---

## 📄 5. Tài liệu & Kế hoạch 2 Tuần
Vui lòng tham khảo file [PLAN_TEAM.md](PLAN_TEAM.md) để xem chi tiết danh sách Task theo từng ngày cho từng thành viên trong nhóm.
