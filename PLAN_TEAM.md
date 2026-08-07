# 📋 Bảng Phân Công Công Việc Nhóm (2-Week Task Assignment)

## 🗓️ TUẦN 1: BÓC TÁCH MODULE & KHÔNG GIAN DỮ LIỆU

### 👤 TECH LEAD (BẠN) - Core AI Engine
- [ ] **Task 1.1**: Đóng gói `src/core/image_encoder.py` (CLIP Vision Backbone).
- [ ] **Task 1.2**: Hoàn thiện `src/core/matcher.py` (Nhân ma trận Cosine Similarity & Softmax).
- [ ] **Task 1.3**: Viết `src/core/pipeline.py` đóng gói class `GeoCLIPService` làm API Service chung cho team.

### 👤 GIS ENGINEER (BẠN ĐỒNG HÀNH GIS) - Phép chiếu & Dữ liệu GPS
- [ ] **Task 2.1**: Hoàn thiện `src/gis/projections.py` (Equal Earth Projection) và `rff_layers.py` (Random Fourier Features).
- [ ] **Task 2.2**: Thu thập và bổ sung 100-200 địa danh Việt Nam vào `data/vietnam_landmarks.csv`.
- [ ] **Task 2.3**: Viết Notebook EDA `notebooks/1_geospatial_eda.ipynb` phân tích độ phủ dữ liệu GPS.

### 👤 FRONTEND TEAM (3 BẠN UI) - Web App & Bản đồ
- [ ] **Task 3.1**: Hoàn thiện `src/viz/map_builder.py` (Custom icon Marker, Popup % xác suất).
- [ ] **Task 3.2**: Hoàn thiện giao diện Streamlit trong `app.py` (Drag & drop ảnh, Top-K slider, responsive layout).

---

## 🗓️ TUẦN 2: TÍCH HỢP, ĐO ĐẠC SAI SỐ & NỘP BÀI

### 👤 TECH LEAD (BẠN)
- [ ] Tích hợp `vietnam_landmarks.csv` vào Gallery chung.
- [ ] Tối ưu bộ nhớ RAM với `@st.cache_resource` khi load model.
- [ ] Tổng hợp code, review pull requests và hoàn thiện slide thuyết trình.

### 👤 GIS ENGINEER
- [ ] Triển khai `src/gis/distance_metrics.py` (Đo sai số khoảng cách Geodesic Error km).
- [ ] Viết Notebook `notebooks/2_model_evaluation.ipynb` vẽ đường cong sai số chính xác (Acc@1km, Acc@25km, Acc@200km...).

### 👤 FRONTEND TEAM
- [ ] Thêm hiệu ứng biểu đồ phân bố xác suất bằng `Altair` / `Plotly`.
- [ ] Kiểm thử toàn bộ giao diện Web trên màn hình máy tính & thiết bị di động.
