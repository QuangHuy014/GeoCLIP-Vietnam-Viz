# TÓM TẮT KHAI BÁO SỬ DỤNG TRÍ TUỆ NHÂN TẠO (AI SUMMARY)
*(Tài liệu tóm tắt đính kèm Báo cáo Đồ án Môn học)*

---

### 1. Thông Tin Thành Viên & Dự Án
* **Họ và tên:** Đoàn Hữu Hàn  
* **Username Git / GitHub:** `HuuHan12`  
* **Email commit:** `dnhan.a7.c3tqcap@gmail.com` / `164132661+HuuHan12@users.noreply.github.com`  
* **Tên Đồ án:** Hệ thống nhận diện địa danh Việt Nam & Đo đạc sai số GIS (`GeoCLIP-Vietnam-Viz` / `IE221-DoAnPython`)  
* **Vai trò trong nhóm:** GIS & Geospatial Engineer (Phụ trách Phân hệ Trắc địa Không gian & Đo đạc Sai số GIS) & Fullstack / Backend Developer (Phụ trách Phân hệ Thống kê Quản trị, Phê duyệt Thanh toán Bảo mật 2 chiều qua Telegram Bot).  

---

### 2. Tóm Lược Mức Độ & Tính Chất Sử Dụng AI

| Tiêu chí | Nội dung chi tiết |
| :--- | :--- |
| **Công cụ AI sử dụng** | Trợ lý Lập trình AI (AI Coding Assistant trong IDE) & Mô hình học sâu lõi `GeoCLIP` (`CLIP ViT-L/14` + `RFF Location Encoder`). |
| **Tính chất sử dụng** | Đóng vai trò **Kênh tham vấn chuyên môn & Hỏi đáp kỹ thuật (Technical Consultation / Pair-Programming)** tương tự việc tra cứu tài liệu đặc tả chuẩn quốc tế hoặc thảo luận giải pháp với chuyên gia. |
| **Tỷ lệ đóng góp thực tế** | **Toàn bộ kiến trúc hệ thống, giải thuật toán học, luồng xử lý và 100% mã nguồn do thành viên tự thiết kế, tự viết và tự kiểm thử.** AI chỉ hỗ trợ tham vấn công thức giải tích và phân tích nguyên nhân lỗi khi gặp vướng mắc kỹ thuật. |
| **Các nhóm kỹ năng (Skills) sử dụng** | 1. **Toán học không gian:** Tham vấn giải tích phép chiếu phẳng bảo toàn diện tích Equal Earth Projection và kỹ thuật xử lý Tensor PyTorch không sinh lỗi NaN/Inf.<br>2. **Mã hóa sóng đa tần số RFF:** Tham vấn cấu trúc lớp Fourier Gaussian Encoding đa tỉ lệ ($\sigma \in \{1, 16, 256\}$) để giải quyết hiện tượng Spectral Bias của mạng nơ-ron.<br>3. **Trắc địa & Đo đạc sai số:** Tham vấn công thức Haversine Great-Circle mặt cầu ($R=6371.0088\text{ km}$), đối chiếu với Geodesic WGS-84 và xây dựng thang đo độ phân giải chuẩn học thuật quốc tế Acc@K (ICCV Standard).<br>4. **Kiến trúc Backend & Bảo mật:** Tham vấn thiết kế giải thuật Zero-filling bù ngày trống, xuất Excel qua luồng RAM `io.BytesIO` và xây dựng giải pháp phê duyệt thanh toán 2 chiều bảo mật qua Telegram Bot. |

---

### 3. Các Đóng Góp Mã Nguồn Chính Của Thành Viên `HuuHan12`
*(Được xác thực 100% qua lịch sử Git commit `git log --author="HuuHan12" --stat`)*

1. **Phân hệ Phép chiếu Bản đồ & Mã hóa Không gian Đa Tần số RFF (Task 2.1):**
   * Hiện thực hóa phép chiếu giải tích **Equal Earth Projection** chuyển đổi tọa độ WGS-84 sang mặt phẳng 2D, hỗ trợ cả tensor 1D và lô tensor 2D ([`src/gis/projections.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/projections.py)).
   * Xây dựng lớp mã hóa sóng tần số ngẫu nhiên Fourier đa thang đo `GaussianEncoding` và `MultiScaleGaussianEncoding` ([`src/gis/rff_layers.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/rff_layers.py)).
   * Triển khai mạng nơ-ron không gian Standalone Location Encoder có bộ kiểm tra dữ liệu chống `NaN`/`Inf` ([`src/gis/location_encoder.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/location_encoder.py)).
   * Xây dựng bộ Unit Test kiểm thử phép chiếu và tính tất định của ma trận tần số $B$ ([`tests/test_gis_module.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/tests/test_gis_module.py), [`test_demo.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/test_demo.py)).
   * Tích hợp dữ liệu hơn 84,000 điểm POI OpenStreetMap Việt Nam trích xuất từ HOTOSM kèm giấy phép bản quyền ODbL ([`data/hotosm_vnm_...geojson`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/data/hotosm_vnm_points_of_interest_points_geojson.geojson), [`Readme.txt`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/Readme.txt)).

2. **Phân hệ Đo Đạc Sai Số Trắc Địa & Tiêu Chuẩn Acc@K (Task 2.3):**
   * Xây dựng các hàm tính khoảng cách đường cong mặt cầu: `haversine_distance`, `calculate_geodesic_distance`, `calculate_batch_haversine_distance` ([`src/gis/distance_metrics.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/distance_metrics.py)).
   * Hiện thực hóa thang đo chuẩn học thuật quốc tế `compute_distance_accuracy_metrics`: **Acc@1km** (Street-level), **Acc@25km** (City-level), **Acc@200km** (Region-level), **Acc@750km** (Country-level), **Acc@2500km** (Continent-level) kèm sai số trung bình `mean_error_km` và trung vị `median_error_km`.
   * Xây dựng bộ Unit Test kiểm thử sai số thực tế đối chiếu Hà Nội $\leftrightarrow$ TP.HCM (~1130km) và kiểm thử lô Tensor ([`tests/test_distance_metrics.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/tests/test_distance_metrics.py)).

3. **Phân hệ Ứng Dụng Mở Rộng & Bảo Mật Hệ Thống:**
   * **Dashboard Thống kê Quản trị (5 Endpoints):** 4 thẻ KPI Overview, biểu đồ tìm kiếm lọc theo thời gian kèm thuật toán Zero-filling bù ngày trống, Top 10 địa danh, cơ cấu danh mục và chức năng xuất báo cáo Excel đa Sheet qua RAM `io.BytesIO`.
   * **Hệ thống Phê duyệt Thanh toán 2 Chiều qua Telegram Bot:** Giải quyết triệt để rủi ro gian lận khi client tự duyệt, xây dựng tiến trình nền listener trong FastAPI lifespan nhận sự kiện bấm nút Inline Keyboard, tự động kích hoạt gói Pro trong CSDL Supabase.
   * **Hệ thống Danh hiệu & Thành tích Check-in:** Xây dựng API và cơ chế trigger tự động cập nhật tiến độ mở khóa danh hiệu ngay sau mỗi lượt quét ảnh thành công.

---

### 4. Một Số Câu Prompt Tiêu Biểu & Quy Trình Kiểm Chứng Độc Lập

* **Prompt 1 (Toán học Phép chiếu Equal Earth):**  
  > *Câu hỏi của HuuHan12:* "Hãy giải thích công thức giải tích phép chiếu Equal Earth và hướng dẫn viết hàm `equal_earth_projection` trên PyTorch hỗ trợ cả tensor 1D và lô tensor 2D, bắt lỗi biên [-90, 90], [-180, 180] và kiểm tra chống lỗi NaN/Inf."  
  > *Kiểm chứng thực tế:* Tự code `projections.py`, viết unit test kiểm thử tọa độ thực tế TP.HCM, Hà Nội và kiểm tra bắt ngoại lệ khi tọa độ vượt biên (`test_single_coordinate_tphcm`, `test_invalid_latitude_raises`), đạt 100% Pass.

* **Prompt 2 (Đo đạc Sai số Geodesic & Tiêu chuẩn Acc@K):**  
  > *Câu hỏi của HuuHan12:* "Trong bài toán Visual Geo-localization, làm sao đo sai số khoảng cách km giữa Ground Truth và Prediction? Sự khác biệt giữa Haversine và Geodesic? Các bài báo như GeoCLIP (ICCV 2023) tính thang đo Acc@K như thế nào?"  
  > *Kiểm chứng thực tế:* Tự code `distance_metrics.py`, viết test case đối soát khoảng cách Hà Nội - TP.HCM nằm trong khoảng 1100km - 1170km và kiểm tra phân bố % Acc@K trên batch 4 mẫu thực tế.

* **Prompt 3 (Bảo mật Thanh toán & Telegram Bot Approval):**  
  > *Câu hỏi của HuuHan12:* "Client tự xác nhận chuyển khoản VietQR rất dễ gian lận. Dự án không có trang Admin riêng, làm sao dùng Telegram Bot để khi user nộp mã giao dịch thì bot gửi tin nhắn kèm nút Duyệt cho tôi bấm duyệt ngay trên điện thoại?"  
  > *Kiểm chứng thực tế:* Tạo Telegram Bot, cấu hình token, viết service `telegram_bot.py` kèm listener trong FastAPI lifespan; thực hiện chuyển khoản thật $\to$ nhận tin nhắn trên Telegram $\to$ bấm nút duyệt $\to$ CSDL Supabase và modal web tự động chuyển sang gói Pro thành công.

---

### 5. Quy Trình Kiểm Thử 5 Bước

1. **Kiểm tra Cú pháp & Unit Test:** Chạy `python -m unittest tests/test_gis_module.py` và `python -m unittest tests/test_distance_metrics.py` đạt 100% Pass.
2. **Kiểm tra Tích hợp Hệ thống:** Chạy `python test_vietnam_full.py`, nhận diện chính xác Landmark 81 với sai số $< 25\text{ km}$ (đạt chuẩn City-level Acc@25km) và xuất bản đồ HTML thành công.
3. **Kiểm tra Sạch Conflict Git:** Chạy `git grep "<<<<<<<"` xác nhận không còn ký tự xung đột.
4. **Kiểm thử API Độc lập:** Sử dụng Swagger Docs (`/docs`) và Postman kiểm tra dữ liệu thật từ CSDL Supabase.
5. **Kiểm thử Luồng Phê duyệt Telegram Bot Thực tế:** Bấm duyệt trên ứng dụng Telegram điện thoại và xác minh CSDL Supabase cập nhật trạng thái `completed`.

---

### 6. Cam Kết Đạo Đức Học Thuật

* Thành viên **Đoàn Hữu Hàn** (`HuuHan12`) cam đoan toàn bộ nội dung khai báo trên phản ánh chính xác 100% quá trình làm việc độc lập của bản thân dựa trên lịch sử Git commit của dự án.
* Công cụ AI được sử dụng đúng chuẩn mực với vai trò **tham vấn kỹ thuật**; mọi dòng mã nguồn đưa vào đồ án đều được thành viên tự tay viết, hiểu rõ bản chất và kiểm thử hoàn chỉnh.
