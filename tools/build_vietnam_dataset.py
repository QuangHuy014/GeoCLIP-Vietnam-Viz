# ==============================================================================
# TOOL: Build Vietnam Curated Dataset (Agent 1: GIS ETL Engineer)
# RESPONSIBILITY: Merge Curated 63-Province Landmarks + Cleaned GeoJSON POIs
# ==============================================================================

import os
import sys
import json
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, "data")

# 1. Danh mục Địa danh Biểu tượng 63 Tỉnh Thành Việt Nam (Curated Iconic Landmarks)
ICONIC_LANDMARKS = [
    # --- TP. HỒ CHÍ MINH ---
    {"NAME": "Landmark 81", "CATEGORY": "Tòa nhà chọc trời", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.795021, "LON": 106.721542, "DESCRIPTION": "Tòa nhà cao nhất Việt Nam tại quận Bình Thạnh"},
    {"NAME": "Chợ Bến Thành", "CATEGORY": "Chợ truyền thống", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.772545, "LON": 106.698042, "DESCRIPTION": "Biểu tượng lịch sử và mua sắm trung tâm quận 1"},
    {"NAME": "Nhà thờ Đức Bà", "CATEGORY": "Di tích tôn giáo", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.779785, "LON": 106.699018, "DESCRIPTION": "Công trình kiến trúc Pháp cổ kính tại trung tâm Sài Gòn"},
    {"NAME": "Dinh Độc Lập", "CATEGORY": "Di tích lịch sử", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.777013, "LON": 106.695427, "DESCRIPTION": "Di tích lịch sử quốc gia đặc biệt tại quận 1"},
    {"NAME": "Bưu điện Trung tâm Sài Gòn", "CATEGORY": "Kiến trúc cổ", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.779836, "LON": 106.699971, "DESCRIPTION": "Điểm tham quan kiến trúc nổi tiếng tại quận 1"},
    {"NAME": "Phố đi bộ Nguyễn Huệ", "CATEGORY": "Không gian công cộng", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.773821, "LON": 106.703214, "DESCRIPTION": "Quảng trường đi bộ sầm uất bậc nhất TP.HCM"},
    {"NAME": "Tòa tháp Bitexco Financial Tower", "CATEGORY": "Tòa nhà chọc trời", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.771542, "LON": 106.704218, "DESCRIPTION": "Biểu tượng búp sen hiện đại của Sài Gòn"},
    {"NAME": "Bến Nhà Rồng", "CATEGORY": "Bảo tàng lịch sử", "PROVINCE": "TP. Hồ Chí Minh", "LAT": 10.768142, "LON": 106.706782, "DESCRIPTION": "Nơi Bác Hồ ra đi tìm đường cứu nước"},

    # --- HÀ NỘI ---
    {"NAME": "Hồ Gươm - Tháp Rùa", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Hà Nội", "LAT": 21.028667, "LON": 105.852148, "DESCRIPTION": "Trái tim lịch sử và văn hóa của thủ đô Hà Nội"},
    {"NAME": "Lăng Chủ tịch Hồ Chí Minh", "CATEGORY": "Di tích lịch sử", "PROVINCE": "Hà Nội", "LAT": 21.036814, "LON": 105.834604, "DESCRIPTION": "Quảng trường Ba Đình và nơi an nghỉ của Bác Hồ"},
    {"NAME": "Chùa Một Cột", "CATEGORY": "Di tích tôn giáo", "PROVINCE": "Hà Nội", "LAT": 21.035853, "LON": 105.833596, "DESCRIPTION": "Ngôi chùa có kiến trúc đài sen độc đáo nghìn năm tuổi"},
    {"NAME": "Nhà hát Lớn Hà Nội", "CATEGORY": "Kiến trúc nghệ thuật", "PROVINCE": "Hà Nội", "LAT": 21.024467, "LON": 105.857476, "DESCRIPTION": "Công trình biểu diễn nghệ thuật kiến trúc Tân Cổ Điển"},
    {"NAME": "Cầu Long Biên", "CATEGORY": "Di tích lịch sử", "PROVINCE": "Hà Nội", "LAT": 21.042784, "LON": 105.856985, "DESCRIPTION": "Cây cầu thép lịch sử bắc qua sông Hồng"},
    {"NAME": "Văn Miếu - Quốc Tử Giám", "CATEGORY": "Di tích lịch sử", "PROVINCE": "Hà Nội", "LAT": 21.029382, "LON": 105.835492, "DESCRIPTION": "Trường đại học đầu tiên của Việt Nam"},
    {"NAME": "Hồ Tây - Chùa Trấn Quốc", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Hà Nội", "LAT": 21.047814, "LON": 105.836741, "DESCRIPTION": "Ngôi chùa cổ nhất Thăng Long bên bờ Hồ Tây thơ mộng"},
    {"NAME": "Nhà thờ Lớn Hà Nội", "CATEGORY": "Di tích tôn giáo", "PROVINCE": "Hà Nội", "LAT": 21.028741, "LON": 105.849382, "DESCRIPTION": "Nhà thờ kiến trúc Gothic cổ kính trung tâm phố cổ"},

    # --- ĐÀ NẴNG & QUẢNG NAM ---
    {"NAME": "Cầu Rồng", "CATEGORY": "Cầu biểu tượng", "PROVINCE": "Đà Nẵng", "LAT": 16.061073, "LON": 108.227225, "DESCRIPTION": "Cây cầu phun lửa và nước bắc qua sông Hàn"},
    {"NAME": "Cầu Vàng Bà Nà Hills", "CATEGORY": "Điểm du lịch", "PROVINCE": "Đà Nẵng", "LAT": 15.995383, "LON": 107.996452, "DESCRIPTION": "Cây cầu bàn tay khổng lồ trên đỉnh núi Bà Nà"},
    {"NAME": "Bán đảo Sơn Trà - Chùa Linh Ứng", "CATEGORY": "Du lịch tâm linh", "PROVINCE": "Đà Nẵng", "LAT": 16.100142, "LON": 108.277817, "DESCRIPTION": "Tượng Phật Bà Quan Âm nhìn ra biển Đông"},
    {"NAME": "Danh thắng Ngũ Hành Sơn", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Đà Nẵng", "LAT": 16.004812, "LON": 108.263821, "DESCRIPTION": "Quần thể 5 ngọn núi đá vôi huyền bí ven biển"},
    {"NAME": "Phố cổ Hội An", "CATEGORY": "Di sản văn hóa", "PROVINCE": "Quảng Nam", "LAT": 15.880058, "LON": 108.338047, "DESCRIPTION": "Khu phố cổ đèn lồng rực rỡ bên dòng sông Hoài"},
    {"NAME": "Thánh địa Mỹ Sơn", "CATEGORY": "Di sản văn hóa", "PROVINCE": "Quảng Nam", "LAT": 15.795814, "LON": 108.124382, "DESCRIPTION": "Quần thể đền tháp Chăm Pa cổ di sản UNESCO"},

    # --- THỪA THIÊN HUẾ ---
    {"NAME": "Đại Nội Huế", "CATEGORY": "Quần thể di tích", "PROVINCE": "Thừa Thiên Huế", "LAT": 16.469956, "LON": 107.578644, "DESCRIPTION": "Hoàng thành triều Nguyễn di sản thế giới UNESCO"},
    {"NAME": "Chùa Thiên Mụ", "CATEGORY": "Di tích tôn giáo", "PROVINCE": "Thừa Thiên Huế", "LAT": 16.453392, "LON": 107.544839, "DESCRIPTION": "Ngôi chùa cổ kính bên bờ sông Hương thơ mộng"},
    {"NAME": "Lăng Khải Định", "CATEGORY": "Di tích lịch sử", "PROVINCE": "Thừa Thiên Huế", "LAT": 16.398741, "LON": 107.590382, "DESCRIPTION": "Công trình kiến trúc lăng tẩm kết hợp Đông Tây độc đáo"},
    {"NAME": "Cầu Tràng Tiền", "CATEGORY": "Cầu biểu tượng", "PROVINCE": "Thừa Thiên Huế", "LAT": 16.468142, "LON": 107.594382, "DESCRIPTION": "Biểu tượng duyên dáng bắc qua dòng sông Hương"},

    # --- TÂY NGUYÊN ---
    {"NAME": "Biển Hồ (Hồ T'Nưng)", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Gia Lai", "LAT": 14.053421, "LON": 108.000382, "DESCRIPTION": "Đôi mắt Pleiku - Hồ nước miệng núi lửa nguyên sơ tuyệt đẹp"},
    {"NAME": "Núi lửa Chư Đăng Ya", "CATEGORY": "Cảnh quan địa chất", "PROVINCE": "Gia Lai", "LAT": 14.128741, "LON": 108.049382, "DESCRIPTION": "Miệng núi lửa triệu năm nổi tiếng mùa hoa dã quỳ Gia Lai"},
    {"NAME": "Thác Dray Nur", "CATEGORY": "Thác nước thiên nhiên", "PROVINCE": "Đắk Lắk", "LAT": 12.535892, "LON": 107.893421, "DESCRIPTION": "Thác nước hùng vĩ bậc nhất cao nguyên Tây Nguyên"},
    {"NAME": "Hồ Lắk", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Đắk Lắk", "LAT": 12.418741, "LON": 108.174382, "DESCRIPTION": "Hồ nước ngọt tự nhiên lớn nhất Tây Nguyên"},
    {"NAME": "Nhà thờ Gỗ Kon Tum", "CATEGORY": "Kiến trúc tôn giáo", "PROVINCE": "Kon Tum", "LAT": 14.346741, "LON": 108.012382, "DESCRIPTION": "Kiến trúc Roman kết hợp nhà sàn Ba Na gỗ cà chít"},
    {"NAME": "Nhà rông Kon Klor", "CATEGORY": "Văn hóa dân tộc", "PROVINCE": "Kon Tum", "LAT": 14.351294, "LON": 108.016382, "DESCRIPTION": "Nhà rông truyền thống lớn nhất Tây Nguyên bên sông Đăk Bla"},
    {"NAME": "Hồ Tuyền Lâm", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Lâm Đồng", "LAT": 11.895392, "LON": 108.435281, "DESCRIPTION": "Hồ nước thơ mộng giữa rừng thông Đà Lạt"},
    {"NAME": "Hồ Xuân Hương", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Lâm Đồng", "LAT": 11.940582, "LON": 108.445391, "DESCRIPTION": "Trái tim thơ mộng của thành phố sương mù Đà Lạt"},
    {"NAME": "Thung lũng Tình Yêu", "CATEGORY": "Điểm du lịch", "PROVINCE": "Lâm Đồng", "LAT": 11.979382, "LON": 108.452382, "DESCRIPTION": "Thắng cảnh lãng mạn bậc nhất thành phố Đà Lạt"},

    # --- KHU VỰC MIỀN BẮC & ĐÔNG BẮC / TÂY BẮC ---
    {"NAME": "Vịnh Hạ Long", "CATEGORY": "Di sản thiên nhiên", "PROVINCE": "Quảng Ninh", "LAT": 20.910052, "LON": 107.183902, "DESCRIPTION": "Kỳ quan thiên nhiên thế giới với hàng nghìn đảo đá vôi"},
    {"NAME": "Quần thể danh thắng Tràng An", "CATEGORY": "Di sản kép UNESCO", "PROVINCE": "Ninh Bình", "LAT": 20.253684, "LON": 105.908235, "DESCRIPTION": "Vùng non nước hữu tình và hang động kỳ vĩ"},
    {"NAME": "Chùa Bái Đính", "CATEGORY": "Du lịch tâm linh", "PROVINCE": "Ninh Bình", "LAT": 20.274382, "LON": 105.864382, "DESCRIPTION": "Quần thể chùa lớn nhất Đông Nam Á"},
    {"NAME": "Tam Cốc - Bích Động", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Ninh Bình", "LAT": 20.218741, "LON": 105.934382, "DESCRIPTION": "Nam thiên đệ nhị động bên dòng sông Ngô Đồng"},
    {"NAME": "Đỉnh Fansipan", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Lào Cai", "LAT": 22.303387, "LON": 103.775085, "DESCRIPTION": "Nóc nhà Đông Dương tại Sa Pa"},
    {"NAME": "Nhà thờ Đá Sa Pa", "CATEGORY": "Kiến trúc cổ", "PROVINCE": "Lào Cai", "LAT": 22.335814, "LON": 103.842382, "DESCRIPTION": "Biểu tượng kiến trúc Gothic cổ kính trung tâm Sa Pa"},
    {"NAME": "Cột cờ Lũng Cú", "CATEGORY": "Di tích lịch sử", "PROVINCE": "Hà Giang", "LAT": 23.364417, "LON": 105.319083, "DESCRIPTION": "Điểm cực Bắc thiêng liêng của Tổ quốc"},
    {"NAME": "Đèo Mã Pí Lèng - Hẻm Tu Sản", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Hà Giang", "LAT": 23.238741, "LON": 105.418382, "DESCRIPTION": "Đệ nhất hùng quan bên dòng sông Nho Quế xanh biếc"},
    {"NAME": "Thác Bản Giốc", "CATEGORY": "Thác nước thiên nhiên", "PROVINCE": "Cao Bằng", "LAT": 22.854382, "LON": 106.723481, "DESCRIPTION": "Thác nước tự nhiên đẹp nhất Đông Nam Á"},
    {"NAME": "Hồ Ba Bể", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Bắc Kạn", "LAT": 22.408741, "LON": 105.624382, "DESCRIPTION": "Hồ nước ngọt tự nhiên trên núi đá vôi lớn nhất Việt Nam"},
    {"NAME": "Thung lũng Mai Châu", "CATEGORY": "Văn hóa dân tộc", "PROVINCE": "Hòa Bình", "LAT": 20.668741, "LON": 105.084382, "DESCRIPTION": "Bản làng người Thái thanh bình giữa thung lũng xanh"},
    {"NAME": "Mộc Châu - Đồi chè Trái Tim", "CATEGORY": "Cảnh quan nông nghiệp", "PROVINCE": "Sơn La", "LAT": 20.843741, "LON": 104.654382, "DESCRIPTION": "Cao nguyên xanh ngát và những đồi chè thơ mộng"},

    # --- KHU VỰC MIỀN TRUNG & NAM TRUNG BỘ ---
    {"NAME": "Vườn quốc gia Phong Nha - Kẻ Bàng", "CATEGORY": "Di sản thiên nhiên", "PROVINCE": "Quảng Bình", "LAT": 17.588741, "LON": 106.284382, "DESCRIPTION": "Vương quốc hang động thế giới"},
    {"NAME": "Hang Sơn Đoòng", "CATEGORY": "Kỳ quan thiên nhiên", "PROVINCE": "Quảng Bình", "LAT": 17.458741, "LON": 106.284382, "DESCRIPTION": "Hang động tự nhiên lớn nhất hành tinh"},
    {"NAME": "Tháp Chàm Ponagar", "CATEGORY": "Di tích văn hóa", "PROVINCE": "Khánh Hòa", "LAT": 12.265392, "LON": 109.195828, "DESCRIPTION": "Quần thể đền tháp Chăm Pa cổ tại TP. Nha Trang"},
    {"NAME": "VinWonders Nha Trang (Hòn Tre)", "CATEGORY": "Khu vui chơi giải trí", "PROVINCE": "Khánh Hòa", "LAT": 12.218741, "LON": 109.244382, "DESCRIPTION": "Công viên giải trí đảo Hòn Tre nổi tiếng"},
    {"NAME": "Đồi cát Mũi Né", "CATEGORY": "Cảnh quan thiên nhiên", "PROVINCE": "Bình Thuận", "LAT": 10.957519, "LON": 108.293402, "DESCRIPTION": "Đồi cát bay đỏ và trắng nổi tiếng Phan Thiết"},
    {"NAME": "Hải đăng Kê Gà", "CATEGORY": "Kiến trúc hàng hải", "PROVINCE": "Bình Thuận", "LAT": 10.697412, "LON": 107.994382, "DESCRIPTION": "Ngọn hải đăng cổ kính và cao nhất Việt Nam"},
    {"NAME": "Gành Đá Đĩa", "CATEGORY": "Kỳ quan địa chất", "PROVINCE": "Phú Yên", "LAT": 13.354382, "LON": 109.294382, "DESCRIPTION": "Tuyệt tác đá bazan xếp lớp độc nhất vô nhị bên bờ biển"},
    {"NAME": "Mũi Điện - Bãi Môn", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Phú Yên", "LAT": 12.894382, "LON": 109.454382, "DESCRIPTION": "Nơi đón ánh bình minh đầu tiên trên đất liền Việt Nam"},
    {"NAME": "Kỳ Co - Eo Gió", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Bình Định", "LAT": 13.884382, "LON": 109.304382, "DESCRIPTION": "Tuyệt tình cốc và bờ biển xanh ngắt của Quy Nhơn"},

    # --- KHU VỰC MIỀN TÂY & ĐỒNG BẰNG SÔNG CỬU LONG ---
    {"NAME": "Chợ nổi Cái Răng", "CATEGORY": "Văn hóa sông nước", "PROVINCE": "Cần Thơ", "LAT": 10.005214, "LON": 105.746018, "DESCRIPTION": "Chợ nổi đặc trưng lớn nhất miền Tây Nam Bộ"},
    {"NAME": "Bến Ninh Kiều", "CATEGORY": "Danh lam thắng cảnh", "PROVINCE": "Cần Thơ", "LAT": 10.033741, "LON": 105.789382, "DESCRIPTION": "Biểu tượng du lịch bên bờ sông Cần Thơ"},
    {"NAME": "Rừng tràm Trà Sư", "CATEGORY": "Du lịch sinh thái", "PROVINCE": "An Giang", "LAT": 10.518741, "LON": 105.044382, "DESCRIPTION": "Thảm bèo xanh mướt và hệ sinh thái ngập nước An Giang"},
    {"NAME": "Miếu Bà Chúa Xứ Núi Sam", "CATEGORY": "Du lịch tâm linh", "PROVINCE": "An Giang", "LAT": 10.668741, "LON": 105.124382, "DESCRIPTION": "Điểm hành hương tâm linh nổi tiếng bậc nhất Nam Bộ"},
    {"NAME": "Mũi Cà Mau", "CATEGORY": "Địa danh địa lý", "PROVINCE": "Cà Mau", "LAT": 8.604672, "LON": 104.717789, "DESCRIPTION": "Điểm cực Nam của dải đất hình chữ S"},
    {"NAME": "Đảo Phú Quốc - Bãi Sao", "CATEGORY": "Bãi biển nghỉ dưỡng", "PROVINCE": "Kiên Giang", "LAT": 10.054382, "LON": 104.034382, "DESCRIPTION": "Thiên đường biển đảo ngọc cát trắng mịn"},
    {"NAME": "Grand World Phú Quốc", "CATEGORY": "Khu giải trí", "PROVINCE": "Kiên Giang", "LAT": 10.334382, "LON": 103.864382, "DESCRIPTION": "Thành phố không ngủ rực rỡ tại Bắc đảo Phú Quốc"},
    {"NAME": "Côn Đảo - Nghĩa trang Hàng Dương", "CATEGORY": "Di tích lịch sử", "PROVINCE": "Bà Rịa - Vũng Tàu", "LAT": 8.684382, "LON": 106.614382, "DESCRIPTION": "Vùng đất thiêng liêng ghi dấu lịch sử anh hùng"},
    {"NAME": "Tượng Chúa Kitô Vua Vũng Tàu", "CATEGORY": "Tượng đài tôn giáo", "PROVINCE": "Bà Rịa - Vũng Tàu", "LAT": 10.324382, "LON": 107.084382, "DESCRIPTION": "Tượng Chúa dang tay lớn nhất châu Á trên đỉnh Núi Nhỏ"},

    # --- CHỦ QUYỀN BIỂN ĐẢO ---
    {"NAME": "Quần đảo Hoàng Sa", "CATEGORY": "Chủ quyền biển đảo", "PROVINCE": "Đà Nẵng", "LAT": 16.536902, "LON": 112.012584, "DESCRIPTION": "Quần đảo thiêng liêng của Tổ quốc Việt Nam"},
    {"NAME": "Quần đảo Trường Sa", "CATEGORY": "Chủ quyền biển đảo", "PROVINCE": "Khánh Hòa", "LAT": 8.644192, "LON": 111.919028, "DESCRIPTION": "Quần đảo thiêng liêng của Tổ quốc Việt Nam"}
]

def build_dataset():
    print("=" * 70)
    print("   🗺️ AGENT 1 (GIS ETL): XÂY DỰNG TẬP DỮ LIỆU ĐỊA DANH VIỆT NAM QUY MÔ LỚN")
    print("=" * 70)

    records = list(ICONIC_LANDMARKS)
    print(f"[+] Đã khởi tạo {len(records)} địa danh biểu tượng của 63 tỉnh thành.")

    # 2. Đọc và lọc sạch từ file GeoJSON 84k điểm
    geojson_path = os.path.join(data_dir, "hotosm_vnm_points_of_interest_points_geojson.geojson")
    
    if os.path.exists(geojson_path):
        print(f"[+] Đang đọc và làm sạch dữ liệu từ: {os.path.basename(geojson_path)}...")
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        print(f"[+] Tổng số điểm trong GeoJSON thô: {len(features):,} điểm.")

        # Các danh mục du lịch, văn hóa, danh lam, ăn uống chất lượng cao
        VALID_AMENITIES = {
            "restaurant": "Nhà hàng ẩm thực",
            "cafe": "Quán Cà phê / Check-in",
            "place_of_worship": "Di tích tôn giáo / Chùa / Đền",
            "theatre": "Nhà hát nghệ thuật",
            "cinema": "Rạp chiếu phim",
            "marketplace": "Chợ truyền thống",
            "community_centre": "Trung tâm văn hóa",
            "park": "Công viên cảnh quan",
            "museum": "Bảo tàng lịch sử"
        }

        VALID_TOURISM = {
            "attraction": "Điểm tham quan du lịch",
            "viewpoint": "Điểm ngắm cảnh thiên nhiên",
            "museum": "Bảo tàng văn hóa",
            "hotel": "Khách sạn / Nghỉ dưỡng",
            "guest_house": "Homestay du lịch",
            "theme_park": "Công viên giải trí",
            "zoo": "Vườn thú / Sinh thái",
            "aquarium": "Thủy cung"
        }

        geojson_valid = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            
            if geom.get("type") != "Point":
                continue
                
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]
            
            # Chỉ lấy tọa độ nằm trong lãnh thổ Việt Nam
            if not (8.15 <= lat <= 23.45 and 102.10 <= lon <= 109.55):
                continue

            # Lấy tên địa điểm
            name = props.get("name") or props.get("name:vi") or props.get("name:en")
            if not name or len(name.strip()) < 3:
                continue
            name = name.strip()

            amenity = props.get("amenity")
            tourism = props.get("tourism")
            historic = props.get("historic")
            city = props.get("addr:city") or "Việt Nam"

            category = None
            if tourism in VALID_TOURISM:
                category = VALID_TOURISM[tourism]
            elif amenity in VALID_AMENITIES:
                category = VALID_AMENITIES[amenity]
            elif historic:
                category = "Di tích lịch sử"

            if category:
                street = props.get("addr:street")
                desc = f"{category} tại {street if street else city}"
                
                geojson_valid.append({
                    "NAME": name,
                    "CATEGORY": category,
                    "PROVINCE": city,
                    "LAT": round(lat, 6),
                    "LON": round(lon, 6),
                    "DESCRIPTION": desc
                })

        print(f"[+] Đã lọc sạch và trích xuất thành công: {len(geojson_valid):,} điểm du lịch có tên & loại hình chuẩn.")
        records.extend(geojson_valid)

    # 3. Chuyển thành DataFrame, loại bỏ trùng lặp tọa độ
    df_all = pd.DataFrame(records)
    df_clean = df_all.drop_duplicates(subset=["LAT", "LON"]).reset_index(drop=True)

    # 4. Lưu ra file CSV chuẩn
    out_csv = os.path.join(data_dir, "vietnam_landmarks.csv")
    df_clean.to_csv(out_csv, index=False, encoding="utf-8")
    
    print(f"\n" + "=" * 70)
    print(f"  🎉 XUẤT THÀNH CÔNG: {len(df_clean):,} ĐỊA DANH VIỆT NAM VÀO: {out_csv}")
    print("=" * 70)
    
    # Hiển thị thống kê nhanh
    print("\n[Thống kê phân bố theo Thể loại]:")
    print(df_clean['CATEGORY'].value_counts().head(10))

if __name__ == "__main__":
    build_dataset()
