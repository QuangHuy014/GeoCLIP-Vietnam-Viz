# 📦 HƯỚNG DẪN ĐÓNG GÓI & SỬ DỤNG THƯ VIỆN `geoclip-vietnam`

Tài liệu này hướng dẫn chi tiết cách **build thư viện thành file `.whl`**, cách **cài đặt sang project/máy tính khác**, và **cách gọi các hàm API** trong Python.

---

## 🛠️ PHẦN 1: CÁCH BUILD THƯ VIỆN (DÀNH CHO BẠN)

Mỗi khi bạn cập nhật thêm tính năng hoặc dữ liệu mới, hãy mở Terminal/CMD tại thư mục dự án và chạy:

```cmd
# 1. Cài đặt công cụ build chuẩn của Python
pip install build

# 2. Thực hiện đóng gói thư viện
python -m build
```

Sau khi chạy xong, trong thư mục `dist/` sẽ xuất hiện 2 file:
* 📦 **`dist/geoclip_vietnam-1.0.0-py3-none-any.whl`** (File Wheel cài đặt siêu tốc)
* 📄 **`dist/geoclip_vietnam-1.0.0.tar.gz`** (Mã nguồn nén)

---

## 📥 PHẦN 2: CÁCH CÀI ĐẶT THƯ VIỆN Ở CÁC PROJECT KHÁC

Người khác (hoặc ở một project khác như FastAPI, Django, Robot backend...) có thể cài đặt theo 3 cách:

### Cách 1: Cài đặt trực tiếp từ file `.whl` (Khuyên dùng nội bộ nhóm)
Gửi file `.whl` trong thư mục `dist/` cho bạn bè, sau đó chạy:
```cmd
pip install dist/geoclip_vietnam-1.0.0-py3-none-any.whl
```

### Cách 2: Cài đặt trực tiếp từ GitHub (Chuẩn Open-Source)
```cmd
pip install git+https://github.com/QuangHuy014/GeoCLIP-Vietnam-Viz.git
```

### Cách 3: Cài đặt dạng liên kết nội bộ trên cùng máy (`Editable Mode`)
```cmd
pip install -e D:\uit\ky3\lt_python\GeoCLIP-Vietnam-Viz
```

---

## 💻 PHẦN 3: HƯỚNG DẪN GỌI HÀM TRONG CODE PYTHON (CODE EXAMPLES)

Sau khi `pip install`, ở bất kỳ file Python nào cũng có thể import và sử dụng:

### 🌟 1. Dự đoán vị trí ảnh nhanh nhất (3 dòng code)

```python
from src import GeoCLIPService

# 1. Khởi tạo dịch vụ (mặc định: chế độ địa danh biểu tượng Việt Nam)
service = GeoCLIPService(scope="vietnam_iconic")

# 2. Dự đoán tọa độ cho bức ảnh bất kỳ
results = service.predict("duong_dan_anh.jpg", top_k=5)

# 3. In kết quả Top-1
top1 = results[0]
print(f"Địa danh: {top1['name']} ({top1['province']})")
print(f"Loại hình: {top1['category']}")
print(f"Tọa độ: Vĩ độ {top1['lat']:.6f}, Kinh độ {top1['lon']:.6f}")
print(f"Xác suất: {top1['prob_percent']}%")
print(f"Google Maps: {top1['gmaps_url']}")
```

---

### 🌐 2. Tùy chọn 3 chế độ hoạt động (Scope Selector)

```python
from src import GeoCLIPService

# Chế độ 1: Địa danh biểu tượng 63 tỉnh thành Việt Nam (Nhanh & Chuẩn xác nhất)
service_iconic = GeoCLIPService(scope="vietnam_iconic")

# Chế độ 2: Toàn bộ 26,353 địa điểm POI (Quán cà phê, nhà hàng, khách sạn khắp VN)
service_all = GeoCLIPService(scope="vietnam_all")

# Chế độ 3: Toàn cầu 100,000 tọa độ quốc tế
service_global = GeoCLIPService(scope="global")
```

---

### 🗺️ 3. Tạo Bản đồ Tương tác Leaflet / Folium (HTML)

```python
from src import GeoCLIPService, create_prediction_map

service = GeoCLIPService(scope="vietnam_iconic")
predictions = service.predict("data/images.jpg", top_k=5)

# Tạo và lưu bản đồ tương tác ra file HTML
folium_map = create_prediction_map(
    predictions=predictions,
    ground_truth=(10.795021, 106.721542),  # Tọa độ thật (nếu có để vẽ đường sai số)
    save_html_path="ban_do_du_doan.html"
)

print("Đã xuất bản đồ HTML thành công!")
```

---

### 📐 4. Đo đạc khoảng cách sai số Geodesic & Haversine (Task 2.3)

```python
from src import calculate_geodesic_distance, compute_distance_accuracy_metrics

# 1. Tính khoảng cách đường chim bay giữa 2 điểm bất kỳ (km)
hanoi = (21.0285, 105.8542)
tphcm = (10.7769, 106.7009)

dist_km = calculate_geodesic_distance(hanoi, tphcm, method="haversine")
print(f"Khoảng cách Hà Nội -> TP.HCM: {dist_km:.2f} km")

# 2. Đo tỷ lệ chính xác theo các ngưỡng bán kính Acc@K (1km, 25km, 200km, 750km)
targets = [(10.795, 106.721), (21.028, 105.834)]
predictions = [(10.790, 106.725), (21.030, 105.830)]

metrics = compute_distance_accuracy_metrics(targets, predictions, thresholds_km=[1, 25, 200, 750])
print("Chỉ số Acc@K:", metrics)
# Kết quả: {'Acc@1km': 100.0, 'Acc@25km': 100.0, 'mean_error_km': 0.65, ...}
```

---

## 📖 PHẦN 4: BẢNG TRA CỨU HÀM API (API REFERENCE)

| Tên Hàm / Lớp | Tham số chính | Kiểu trả về | Mô tả chức năng |
| :--- | :--- | :---: | :--- |
| `GeoCLIPService` | `scope="vietnam_iconic" \| "vietnam_all" \| "global"` | `Object` | Khởi tạo mô hình AI và nạp bộ nhớ đệm ma trận tọa độ. |
| `service.predict()` | `image_input (str \| PIL.Image), top_k=5` | `List[Dict]` | Dự đoán Top-K tọa độ kèm Rich Metadata (% Xác suất, Tỉnh/Thành, Mô tả, URL). |
| `create_prediction_map()` | `predictions, ground_truth=None, save_html_path=None` | `folium.Map` | Tạo bản đồ tương tác Leaflet có ghim màu, vòng tròn xác suất và đường nối sai số. |
| `calculate_geodesic_distance()`| `point1=(lat, lon), point2=(lat, lon), method="haversine"` | `float` | Tính khoảng cách đường chim bay chính xác giữa 2 điểm (km). |
| `compute_distance_accuracy_metrics()` | `targets, predictions, thresholds_km=[1, 25, 200, 750]` | `Dict` | Đánh giá độ chính xác `Acc@1km`, `Acc@25km`, `Acc@200km`, `Mean Error`. |
