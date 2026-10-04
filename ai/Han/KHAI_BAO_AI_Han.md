# BẢN KHAI BÁO SỬ DỤNG TRÍ TUỆ NHÂN TẠO (AI DECLARATION REPORT)

> **Môn học:** Lập trình Python / Phân tích và Trực quan hóa Dữ liệu  
> **Dự án:** Hệ thống Nhận diện Địa danh Việt Nam & Đo đạc Sai số Không gian GIS (`GeoCLIP-Vietnam-Viz` / `IE221-DoAnPython`)  
> **Thành viên thực hiện:** Đoàn Hữu Hàn  
> **Username Git / GitHub:** `HuuHan12`  
> **Email commit:** `dnhan.a7.c3tqcap@gmail.com` / `164132661+HuuHan12@users.noreply.github.com`  
> **Vai trò đảm nhiệm trong nhóm:** GIS & Geospatial Engineer (Phân hệ Trắc địa Không gian & Đo đạc Sai số GIS) & Fullstack / Backend Developer (Phân hệ API Thống kê, Phê duyệt Thanh toán Bảo mật qua Telegram Bot)  
> **Ngày lập báo cáo:** 04/10/2026  

---

## 1. Danh Mục Công Cụ AI Đã Sử Dụng

1. **Trợ lý Lập trình AI (AI Coding Assistant):**
   * **Môi trường hoạt động:** Tích hợp trực tiếp trong môi trường phát triển mã nguồn cục bộ (Local IDE).
   * **Bản chất vai trò:** Đóng vai trò như một **Chuyên gia Tham vấn Kỹ thuật (Technical Advisor / Pair-Programming Consultant)** — hỗ trợ giải thích công thức toán học không gian giải tích, tra cứu cú pháp thư viện trắc địa/PyTorch, gợi ý giải thuật xử lý dữ liệu và hỗ trợ rà soát lỗi logic (debugging).
   * **Nguyên tắc kiểm soát:** 100% mã nguồn được đưa vào dự án đều do thành viên `HuuHan12` tự tay đọc hiểu, gõ code, tinh chỉnh, kiểm thử và chịu trách nhiệm độc lập.

2. **Mô hình Học Sâu Lõi trong Sản phẩm (Product Core Deep Learning Model):**
   * **Tên mô hình / Thư viện:** `geoclip-vietnam` (dựa trên kiến trúc mạng nơ-ron đa phương thức **GeoCLIP - ICCV 2023** kết hợp mô hình thị giác **OpenAI CLIP ViT-L/14** và mạng nơ-ron không gian **Location Encoder** sử dụng sóng tần số ngẫu nhiên RFF).
   * **Vai trò trong sản phẩm:** Trích xuất vector đặc trưng thị giác từ ảnh đầu vào và so khớp cosine similarity với ma trận tọa độ địa danh Việt Nam để dự đoán vị trí địa lý.

---

## 2. Mục Đích Sử Dụng AI & Các Nhóm Kỹ Năng Kỹ Thuật (Skills)

Quá trình sử dụng AI của thành viên `HuuHan12` được giới hạn nghiêm ngặt trong 5 nhóm mục đích kỹ thuật cụ thể:

### 🔹 Kỹ năng 1: Toán học Không gian & Phép chiếu Bản đồ (Geospatial Projection Skill)
* **Vấn đề gặp phải:** Tọa độ GPS chuẩn WGS-84 hình cầu $(Lat, Lon)$ nếu đưa trực tiếp vào mạng nơ-ron sẽ gây méo mó nghiêm trọng về tỷ lệ diện tích giữa các vùng địa lý (đặc biệt là vùng cực so với xích đạo).
* **Mục đích tham vấn AI:** Tìm hiểu cơ chế giải tích của phép chiếu bảo toàn diện tích **Equal Earth Projection** (sáng chế bởi Šavrič et al., 2018), các hệ số đa thức $A_1, A_2, A_3, A_4$, cách chuyển đổi vĩ độ sang vĩ độ tham số $\theta$ và tối ưu hóa xử lý song song trên PyTorch Tensor (hỗ trợ cả tensor 1D và lô tensor 2D), đảm bảo không sinh ra giá trị bất định `NaN` hoặc `Inf`.

### 🔹 Kỹ năng 2: Sóng Tần Số Ngẫu Nhiên Fourier (Random Fourier Features - RFF Skill)
* **Vấn đề gặp phải:** Mạng nơ-ron truyền thống mắc hiện tượng **Spectral Bias** (ưu tiên học các hàm tần số thấp, khó nắm bắt sự thay đổi tọa độ địa lý ở quy mô chi tiết).
* **Mục đích tham vấn AI:** Tham vấn cấu trúc lớp `GaussianEncoding` và `MultiScaleGaussianEncoding` biến đổi tọa độ 2D sang không gian sóng đa thang đo $\gamma(v) = [\cos(2\pi B v), \sin(2\pi B v)]$, cách lựa chọn các mốc độ lệch chuẩn $\sigma \in \{2^0, 2^4, 2^8\} = \{1.0, 16.0, 256.0\}$ tương ứng với 3 cấp độ phân giải (toàn cầu/châu lục, quốc gia, khu vực địa danh), và cơ chế cố định ma trận tần số $B$ (`requires_grad=False`) để giữ tính tất định khi suy luận.

### 🔹 Kỹ năng 3: Đo Đạc Sai Số Trắc Địa & Tiêu Chuẩn Acc@K (GIS Error Metrics Skill)
* **Vấn đề gặp phải:** Cần đo lường chính xác khoảng cách sai lệch thực tế giữa vị trí người chụp ảnh (Ground Truth) và vị trí mô hình AI dự đoán (Prediction).
* **Mục đích tham vấn AI:** Tra cứu công thức mặt cầu **Haversine Great-Circle Distance** với bán kính Trái Đất chuẩn $R = 6371.0088\text{ km}$, kỹ thuật kẹp biên `math.atan2` chống lỗi dấu phẩy động; đối chiếu với công thức Geodesic trên elipsoid WGS-84 của thư viện `geopy`; đồng thời xây dựng hàm đánh giá tỷ lệ chính xác theo thang đo chuẩn học thuật quốc tế **Acc@K** (`Acc@1km`, `Acc@25km`, `Acc@200km`, `Acc@750km`, `Acc@2500km`).

### 🔹 Kỹ năng 4: Xử Lý & Làm Sạch Dữ Liệu Lớn Địa Lý (GIS ETL Skill)
* **Vấn đề gặp phải:** File GeoJSON thô từ OpenStreetMap / HOTOSM chứa hơn 84,000 điểm với nhiều định dạng lộn xộn, điểm nằm ngoài biên giới Việt Nam, hoặc thiếu tên địa danh.
* **Mục đích tham vấn AI:** Tra cứu cú pháp Python tối ưu để parse GeoJSON theo dòng, lọc bounding box lãnh thổ Việt Nam ($Lat \in [8.15, 23.45]$, $Lon \in [102.10, 109.55]$), ánh xạ các thẻ `tourism`, `amenity`, `historic` sang danh mục tiếng Việt chuẩn và loại bỏ tọa độ trùng lặp.

### 🔹 Kỹ năng 5: Kiến Trúc API Thống Kê & Phê Duyệt Thanh Toán 2 Chiều qua Telegram Bot
* **Vấn đề gặp phải:** Dự án cần Dashboard quản trị phân tích số liệu tìm kiếm và tính năng nâng cấp gói Pro bằng chuyển khoản VietQR, nhưng hệ thống không có trang Admin riêng để duyệt tiền thủ công; việc để Client tự bấm xác nhận chuyển khoản tiềm ẩn nguy cơ gian lận nghiêm trọng.
* **Mục đích tham vấn AI:** Thiết kế quy trình phê duyệt bảo mật 2 chiều qua Telegram Bot: API tiếp nhận mã giao dịch đối soát (`transaction_ref`), kích hoạt `BackgroundTasks` gửi thông báo kèm Inline Keyboard (`[✅ Phê duyệt ngay]`, `[❌ Từ chối]`) về Telegram của quản trị viên; tiến trình nền `start_telegram_bot_listener` (FastAPI lifespan) lắng nghe callback query, xử lý `allowed_updates` nhận sự kiện bấm nút, tự động kích hoạt gói Pro trong CSDL Supabase và gửi thông báo cho người dùng.

---

## 3. Nhật Ký Prompt Thực Tế, Phản Hồi của AI & Quy Trình Kiểm Chứng

Dưới đây là các câu lệnh (prompts) thực tế tiêu biểu minh họa cho quá trình thành viên `HuuHan12` làm việc cùng AI:

---

### 🔹 Prompt 1: Về Phép Chiếu Bản Đồ Equal Earth & Chống Lỗi Số Học Trên Tensor
> **Bối cảnh:** Thành viên `HuuHan12` nhận nhiệm vụ Task 2.1 xây dựng tệp `src/gis/projections.py`.  
> 
> **Câu Prompt của HuuHan12:**  
> *"Tôi đang triển khai module GIS cho đồ án GeoCLIP Việt Nam bằng PyTorch. Tọa độ đầu vào là vĩ độ và kinh độ WGS84 (độ). Tôi cần chuyển đổi sang hệ tọa độ phẳng 2D bằng phép chiếu Equal Earth để làm đầu vào cho Location Encoder. Hãy giải thích cho tôi công thức giải tích của phép chiếu này, các hằng số đa thức chuẩn, và cách viết hàm `equal_earth_projection` trên PyTorch sao cho nhận được cả tensor 1D `(2,)` lẫn batch 2D `(N, 2)`, đồng thời bắt lỗi nếu vĩ độ vượt quá [-90, 90] hoặc kinh độ ngoài [-180, 180] và kiểm tra chống lỗi NaN/Inf."*  
> 
> **Phản hồi từ AI:**  
> - Cung cấp công thức Equal Earth (Šavrič et al., 2018): đổi độ sang radian; tính vĩ độ tham số $\theta$ thỏa mãn $\sin\theta = \frac{\sqrt{3}}{2} \sin(\phi)$; mẫu số đa thức bậc 8: $D = 3 \cdot (9 A_4 \theta^8 + 7 A_3 \theta^6 + 5 A_2 \theta^4 + 3 A_1 \theta^2 + A_0)$; tính $x = \frac{2\sqrt{3} \lambda \cos\theta}{D}$ và $y = \theta (A_4 \theta^8 + A_3 \theta^6 + A_2 \theta^4 + A_1 \theta^2 + A_0)$ với các hệ số $A_0 \approx 1.340264, A_1 \approx -0.081106, A_2 \approx 0.000893, A_3 \approx 0.003796, A_4 \approx 0.003429$.  
> - Đề xuất kỹ thuật kiểm tra kích thước `dim() == 1` hay `dim() == 2`, dùng `torch.deg2rad` và dùng `torch.clamp` giới hạn $\sin\theta \in [-1.0, 1.0]$ để tránh lỗi `asin` sinh ra `NaN`.  
> - Đề xuất kiểm tra `torch.isnan().any()` và `torch.isinf().any()` trước khi trả về tensor kết quả.  
> 
> **Cách kiểm chứng & thực thi độc lập của HuuHan12:**  
> - Tự viết mã nguồn tệp [`src/gis/projections.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/projections.py) và hoàn thiện thêm trong [`src/gis/location_encoder.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/location_encoder.py).  
> - Viết bộ Unit Test trong [`tests/test_gis_module.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/tests/test_gis_module.py): kiểm thử tọa độ TP.HCM `(10.795, 106.721)`, Hà Nội `(21.028, 105.834)`, kiểm thử batch 3 điểm và kiểm tra ném lỗi `ValueError` khi vĩ độ là `95.0` hoặc kinh độ là `200.0`.  
> - Chạy lệnh `python -m unittest tests/test_gis_module.py` đạt 100% Pass. (Minh chứng tại commit `ac70cf6`).

---

### 🔹 Prompt 2: Về Đo Đạc Sai Số Geodesic, Haversine & Chuẩn Acc@K (Task 2.3)
> **Bối cảnh:** Thành viên `HuuHan12` triển khai module tính khoảng cách sai số `src/gis/distance_metrics.py`.  
> 
> **Câu Prompt của HuuHan12:**  
> *"Trong bài toán Visual Geo-localization, khi mô hình dự đoán ra cặp (Lat, Lon) và tôi có tọa độ thực tế Ground Truth, làm sao để tính khoảng cách sai số chính xác nhất bằng kilômét? Hãy so sánh giữa công thức Haversine và Geodesic (WGS-84). Ngoài ra, các bài báo như GeoCLIP (ICCV 2023) đánh giá độ chính xác theo thang Acc@K như thế nào? Hãy hướng dẫn tôi thiết kế hàm tính toán trả về các chỉ số Acc@1km, Acc@25km, Acc@200km, Acc@750km, Acc@2500km kèm sai số trung bình mean/median."*  
> 
> **Phản hồi từ AI:**  
> - Phân tích: Công thức Haversine giả định Trái Đất hình cầu hoàn hảo với bán kính trung bình $R \approx 6371.0088\text{ km}$, tính toán thuần túy bằng hàm lượng giác (`math.sin`, `math.cos`, `math.atan2`), chạy cực nhanh và không cần cài thư viện ngoài. Trong khi đó, Geodesic (sử dụng thuật toán Karney qua thư viện `geopy`) tính trên mặt cầu dẹt Elipsoid WGS-84, độ chính xác đến từng milimét nhưng tính toán nặng hơn. Khuyên dùng kết hợp: ưu tiên `geopy` nếu môi trường có sẵn, tự động fallback về Haversine nếu không có thư viện ngoài.  
> - Giải thích thang đo phân giải không gian chuẩn Acc@K:
>   * **Acc@1km (Street-level):** Nhận diện chính xác cấp đường phố / địa danh cụ thể.
>   * **Acc@25km (City-level):** Nhận diện đúng cùng thành phố hoặc quận/huyện.
>   * **Acc@200km (Region-level):** Nhận diện đúng cùng vùng miền / tỉnh lân cận.
>   * **Acc@750km (Country-level):** Nhận diện đúng cùng một quốc gia.
>   * **Acc@2500km (Continent-level):** Nhận diện đúng cùng châu lục.
> - Hướng dẫn thuật toán đếm tỷ lệ phần trăm số mẫu có sai số $\le \text{threshold}$ và tính `np.mean()`, `np.median()`.  
> 
> **Cách kiểm chứng & thực thi độc lập của HuuHan12:**  
> - Tự viết mã nguồn tệp [`src/gis/distance_metrics.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/distance_metrics.py) với 4 hàm cốt lõi: `haversine_distance`, `calculate_geodesic_distance`, `calculate_batch_haversine_distance`, `compute_distance_accuracy_metrics`.  
> - Viết bộ kiểm thử chuyên sâu trong [`tests/test_distance_metrics.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/tests/test_distance_metrics.py): kiểm tra khoảng cách giữa 2 điểm trùng nhau bằng 0.0 km; kiểm tra khoảng cách thực tế giữa Hà Nội và TP.HCM nằm trong khoảng $1100\text{ km} - 1170\text{ km}$ (chuẩn thực tế ~1130 km); kiểm tra batch 4 mẫu với kết quả Acc@1km = 50%, Acc@25km = 75%, Acc@2500km = 100%. (Minh chứng tại commit `c067e39`).

---

### 🔹 Prompt 3: Về Thiết Kế Lớp Mã Hóa Sóng Tần Số Fourier Ngẫu Nhiên (RFF)
> **Bối cảnh:** Hiện thực hóa lớp `GaussianEncoding` và `MultiScaleGaussianEncoding` trong `src/gis/rff_layers.py`.  
> 
> **Câu Prompt của HuuHan12:**  
> *"Để Location Encoder không bị bão hòa tần số khi học tọa độ, GeoCLIP sử dụng Random Fourier Features. Tôi muốn viết module `rff_layers.py` chứa lớp `GaussianEncoding` và `MultiScaleGaussianEncoding`. Làm sao để sinh ma trận tần số ngẫu nhiên Gaussian từ độ lệch chuẩn sigma, và làm sao đảm bảo ma trận này không bị cập nhật gradient trong quá trình huấn luyện và giữ nguyên vẹn giữa các lần gọi inference?"*  
> 
> **Phản hồi từ AI:**  
> - Hướng dẫn định nghĩa ma trận trọng số $B \sim \mathcal{N}(0, \sigma^2)$ kích thước `(encoded_size, input_size)`.  
> - Để không bị tính đạo hàm, khai báo ma trận này dưới dạng `nn.Parameter(b, requires_grad=False)` hoặc `self.register_buffer('B', b)`.  
> - Công thức forward: $vp = 2\pi (v \cdot B^T)$, đầu ra kết hợp $[\cos(vp), \sin(vp)]$ để tăng số chiều từ 2 lên $2 \times \text{encoded\_size}$.  
> - Với đa tỉ lệ: lặp qua danh sách `sigmas=[1.0, 16.0, 256.0]`, tạo một `nn.ModuleList` các lớp `GaussianEncoding` và ghép nối (concatenate) đầu ra.  
> 
> **Cách kiểm chứng & thực thi độc lập của HuuHan12:**  
> - Trực tiếp hiện thực tệp [`src/gis/rff_layers.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/rff_layers.py).  
> - Viết test case `test_fourier_matrix_persistence` trong `tests/test_gis_module.py` để xác nhận ma trận $B$ trước và sau nhiều lần gọi hàm `forward()` là hoàn toàn đồng nhất (`torch.equal(layer.B, B_initial)`). (Minh chứng tại commit `ac70cf6`).

---

### 🔹 Prompt 4: Về Bảo Mật Luồng Thanh Toán & Phê Duyệt 2 Chiều Qua Telegram Bot
> **Bối cảnh:** Phát triển phân hệ nâng cấp tài khoản Pro và kiểm soát gian lận thanh toán VietQR.  
> 
> **Câu Prompt của HuuHan12:**  
> *"Trong đồ án của tôi có tính năng nâng cấp gói Pro bằng chuyển khoản ngân hàng VietQR. Hiện tại trên giao diện có nút 'Tôi đã chuyển khoản' để người dùng tự xác nhận. Tôi thấy việc này rất dễ bị gian lận nếu user bấm mà chưa chuyển tiền thật. Nhóm tôi không có trang Admin riêng để duyệt tiền. Có cách nào tích hợp bot Telegram để khi user nộp mã giao dịch thì bot gửi tin nhắn kèm nút Duyệt cho tôi bấm duyệt ngay trên điện thoại không?"*  
> 
> **Phản hồi từ AI:**  
> - Khẳng định việc cho client tự xác nhận là lỗ hổng bảo mật nghiêm trọng.  
> - Đề xuất quy trình duyệt 2 chiều qua Telegram Bot:  
>   1. Client gửi `POST /payments/submit-transfer` kèm mã tham chiếu (`transaction_ref`).  
>   2. Backend đưa tác vụ vào `BackgroundTasks` gọi Telegram Bot API gửi tin nhắn thông báo kèm `inline_keyboard` gồm 2 nút `approve:<order_code>` và `reject:<order_code>`.  
>   3. Viết tiến trình nền Long-Polling (`getUpdates`) trong FastAPI `lifespan` để nhận sự kiện callback query khi Admin bấm nút, sau đó gọi hàm kích hoạt gói Pro trong CSDL Supabase.  
> 
> **Cách kiểm chứng & thực thi độc lập của HuuHan12:**  
> - Tạo bot trên @BotFather, cấu hình `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` vào file môi trường.  
> - Hiện thực hóa `app/services/telegram_bot.py` và tích hợp vào `app/main.py`.  
> - Thử nghiệm thực tế: nộp mã chuyển khoản trên giao diện $\to$ nhận tin nhắn trên Telegram $\to$ bấm nút Phê duyệt $\to$ kiểm tra CSDL Supabase và modal web tự động chuyển sang trạng thái Pro thành công.

---

### 🔹 Prompt 5: Về Rà Soát Sửa Lỗi Click Nút Phê Duyệt Telegram Bot
> **Bối cảnh:** Admin bấm nút "Phê duyệt" trên Telegram nhưng hệ thống không ghi nhận kích hoạt gói Pro.  
> 
> **Câu Prompt của HuuHan12:**  
> *"Kiểm tra cho tôi vì sao khi tôi chuyển khoản ngân hàng xong và đã gửi về telegram nhưng tôi nhấn phê duyệt không thành công, hãy đưa ra plan để sửa trước khi tôi nhấn triển khai nhé."*  
> 
> **Phản hồi từ AI:**  
> - Kiểm tra trạng thái đơn hàng trên CSDL Supabase (xác nhận đơn hàng vẫn ở trạng thái `pending_verification`, chứng tỏ callback handler chưa được kích hoạt).  
> - Kiểm tra cấu hình bot qua API `getWebhookInfo` của Telegram và phát hiện: Telegram đang áp dụng bộ lọc `allowed_updates: ['message', 'edited_message', 'channel_post', 'edited_channel_post']` vốn **không có `callback_query`**, khiến Telegram tự động hủy bỏ sự kiện click nút Inline Keyboard.  
> - Đề xuất phương án sửa: (1) Khai báo rõ ràng `allowed_updates: ["message", "callback_query"]` trong tham số `getUpdates` qua phương thức `POST`; (2) Thoát ký tự HTML (`html.escape`) cho tin nhắn sửa đổi; (3) Bọc hàm xử lý CSDL bằng `asyncio.to_thread`; (4) Thêm supervisor auto-restart trong `main.py`.  
> 
> **Cách kiểm chứng & thực thi độc lập của HuuHan12:**  
> - Trực tiếp cập nhật mã nguồn tệp `telegram_bot.py` và `main.py`.  
> - Bấm nút duyệt trên ứng dụng Telegram điện thoại: bot phản hồi tức thì với alert pop-up và cập nhật tin nhắn sang `✅ ĐÃ ĐƯỢC PHÊ DUYỆT BỞI ADMIN`.

---

## 4. Bảng Kê Chi Tiết File/Code Đóng Góp của Thành Viên `HuuHan12`

Toàn bộ các đóng góp mã nguồn dưới đây được xác thực trực tiếp từ lịch sử Git (`git log --author="HuuHan12" --stat`):

### 4.1. Phân hệ GIS & AI Core trong Thư viện Dự án (`GeoCLIP-Vietnam-Viz`)

| Tệp mã nguồn | Hash Commit | Mục đích & Nội dung kỹ thuật do `HuuHan12` thực hiện |
| :--- | :--- | :--- |
| [`src/gis/projections.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/projections.py) | `ac70cf6`, `c067e39` | Hiện thực hóa công thức giải tích phép chiếu **Equal Earth Projection**, chuyển đổi vĩ độ/kinh độ hình cầu sang mặt phẳng 2D $(x, y)$, xử lý tương thích tensor 1D và lô tensor 2D, bắt lỗi biên $[-90, 90]$ và $[-180, 180]$. |
| [`src/gis/rff_layers.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/rff_layers.py) | `ac70cf6`, `c067e39` | Hiện thực hóa lớp `GaussianEncoding` và `MultiScaleGaussianEncoding` (Random Fourier Features) đa tần số ($\sigma \in \{1, 16, 256\}$), giải quyết hiện tượng Spectral Bias của mạng nơ-ron không gian. |
| [`src/gis/location_encoder.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/location_encoder.py) | `ac70cf6`, `c067e39` | Xây dựng mạng nơ-ron không gian Standalone Location Encoder kết hợp Equal Earth, RFF và mạng MLP nhiều tầng, bổ sung bộ kiểm tra validation chống giá trị vô định `NaN`/`Inf`. |
| [`src/gis/distance_metrics.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/src/gis/distance_metrics.py) | `c067e39` | Xây dựng 4 hàm đo đạc sai số trắc địa: `haversine_distance`, `calculate_geodesic_distance`, `calculate_batch_haversine_distance`, và hàm đánh giá tỷ lệ chính xác chuẩn học thuật quốc tế `compute_distance_accuracy_metrics` (Acc@1km, Acc@25km, Acc@200km, Acc@750km, Acc@2500km). |
| [`tests/test_gis_module.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/tests/test_gis_module.py) | `ac70cf6` | Bộ Unit Test kiểm thử phép chiếu Equal Earth, kiểm tra bắt ngoại lệ khi tọa độ vượt biên, kiểm tra tính tất định và tính bền vững của ma trận tần số $B$ trong lớp RFF. |
| [`tests/test_distance_metrics.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/tests/test_distance_metrics.py) | `c067e39` | Bộ Unit Test kiểm tra tính toán khoảng cách Haversine/Geodesic giữa các điểm mốc chuẩn (Hà Nội $\leftrightarrow$ TP.HCM ~1130km), kiểm thử lô Tensor và tỷ lệ phân bố % Acc@K. |
| [`test_demo.py`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/test_demo.py) | `ac70cf6` | Kịch bản demo kiểm thử nhanh độc lập các hàm biến đổi GIS trong terminal. |
| [`data/hotosm_vnm_...geojson`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/data/hotosm_vnm_points_of_interest_points_geojson.geojson) | `ac70cf6` | Thu thập và tích hợp cơ sở dữ liệu không gian hơn 84,000 điểm POI OpenStreetMap Việt Nam trích xuất từ HOTOSM. |
| [`Readme.txt`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/Readme.txt) | `ac70cf6` | Khai báo bản quyền dữ liệu mở Open Database License (ODbL) theo chuẩn pháp lý của OpenStreetMap. |
| [`README.md`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/README.md) & [`docs/workflow_task2_1_3.jpg`](file:///p:/CITD/HK3/L%E1%BA%ADp%20tr%C3%ACnh%20Python/DuAn/source/docs/workflow_task2_1_3.jpg) | `abe2f5a`, `3c4b214` | Biên soạn tài liệu kỹ thuật về quy trình xử lý không gian trắc địa và cập nhật sơ đồ workflow đồ án. |

### 4.2. Phân hệ Ứng Dụng Mở Rộng Toàn Đồ Án (Fullstack Web & API Services)

| Nhóm chức năng | Tệp liên quan | Mục đích & Nội dung thực hiện |
| :--- | :--- | :--- |
| **API Đo đạc Sai số GIS** | `app/api/gis.py` | API `POST /gis/calculate-error`: Tính toán khoảng cách Geodesic WGS-84, Haversine, góc phương vị Bearing và gán nhãn phân cấp độ chính xác Acc@K cho giao diện người dùng. |
| **Dashboard Thống kê Quản trị (5 Endpoints)** | `app/api/statistics.py`, `app/schemas/statistics.py` | Phát triển toàn diện 5 API: 4 Thẻ KPI Overview (`/overview`), Biểu đồ tần suất tìm kiếm (`/search-trends` với kỹ thuật Zero-filling bù ngày trống), Top 10 địa danh (`/top-places`), Cơ cấu danh mục (`/category-distribution`), Nút Xuất báo cáo Excel 4 Sheets (`/export` qua RAM stream `io.BytesIO`). |
| **Phê duyệt Thanh toán 2 Chiều qua Telegram Bot** | `app/services/telegram_bot.py`, `app/api/payments.py`, `app/main.py` | Thiết kế cơ chế phê duyệt bảo mật: Tiếp nhận mã tham chiếu đối soát (`POST /payments/submit-transfer`), gửi tin nhắn kèm Inline Keyboard về Telegram, listener chạy nền trong FastAPI lifespan tự động duyệt và kích hoạt gói Pro trong Supabase. |
| **Hệ thống Danh hiệu & Thành tích** | `app/api/achievements.py`, `app/api/predict.py` | API tính toán tiến độ mở khóa 7 huy hiệu thành tích check-in, tích hợp trigger tự động cập nhật tiến độ ngay sau mỗi lần scan ảnh thành công. |
| **Giao diện Người dùng Form AI & Thanh toán** | `web/src/components/*` | Component Form đo đạc sai số (`GisErrorTab.jsx`), Modal thanh toán VietQR Napas 247 (`PaymentQRModal.jsx`), custom hook TypeScript (`usePredict.ts`, `geoUtils.ts`). |

---

## 5. Quy Trình Kiểm Thử & Đảm Bảo Chất Lượng (Verification Methods)

Mọi nội dung có sự trợ giúp hoặc tham vấn từ AI đều trải qua quy trình kiểm thử 5 bước nghiêm ngặt:

1. **Kiểm tra Cú pháp & Đơn vị (Unit Testing):**  
   Chạy toàn bộ bộ test tự động của module GIS:
   ```cmd
   python -m unittest tests/test_gis_module.py
   python -m unittest tests/test_distance_metrics.py
   ```
   Kết quả: 100% test cases đều đạt chuẩn (`OK`), không phát sinh lỗi ngoại lệ hoặc giá trị vô định.

2. **Kiểm tra Tích hợp Pipeline (Integration Testing):**  
   Chạy bộ test tích hợp hệ thống:
   ```cmd
   python test_vietnam_full.py
   ```
   Xác nhận mô hình nhận diện chính xác Landmark 81 tại TP.HCM với sai số khoảng cách $< 25\text{ km}$ (đạt chuẩn City-level Acc@25km) và xuất bản đồ Folium thành công.

3. **Kiểm tra Sạch Xung Đột Git (Clean Git History):**  
   Quét toàn bộ repository để đảm bảo không còn bất kỳ conflict marker nào:
   ```cmd
   git grep "<<<<<<<"
   ```

4. **Kiểm thử API Độc lập (Swagger UI / Postman):**  
   Sử dụng Swagger Docs (`/docs`) để kiểm tra toàn bộ các endpoint đo sai số GIS, 5 endpoint thống kê và tải trực tiếp file Excel `.xlsx` 4 Sheets kiểm tra tính toàn vẹn của dữ liệu.

5. **Kiểm thử Luồng Phê duyệt Telegram Bot Thực tế:**  
   Thực hiện giao dịch chuyển khoản trên giao diện web $\to$ nhận thông báo kèm nút bấm trên Telegram Bot $\to$ bấm nút Phê duyệt $\to$ xác nhận CSDL Supabase cập nhật trạng thái `completed`, kích hoạt gói Pro và web tự động cập nhật màn hình thành công.

---

## 6. Đạo Đức Học Thuật & Cam Kết Độc Lập

* Thành viên **Đoàn Hữu Hàn** (`HuuHan12`) cam đoan toàn bộ thông tin khai báo trên là trung thực, phản ánh chính xác quá trình làm việc của bản thân dựa trên lịch sử Git commit của dự án.
* AI được sử dụng đúng mục đích **tham vấn chuyên môn và giải thích kỹ thuật**; không có hành vi sao chép mã nguồn tự động không kiểm soát hoặc nhận vơ phần việc của đồng đội.
* Bản khai báo không chứa bất kỳ khóa bí mật (secret keys), token xác thực hay dữ liệu bảo mật nào của hệ thống.
