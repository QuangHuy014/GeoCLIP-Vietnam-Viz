# ==============================================================================
# ENTRYPOINT: Streamlit Web Dashboard App (Frontend UI Team)
# RESPONSIBILITY: Main Web Dashboard UI Layout, Image Drag & Drop, Maps & Analytics
# ==============================================================================

import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Thêm thư mục gốc vào sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Đảm bảo in được tiếng Việt có dấu
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.core.pipeline import GeoCLIPService
from src.viz.map_builder import create_prediction_map
from src.gis.distance_metrics import calculate_geodesic_distance

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="GeoCLIP Vietnam - Visual Geo-localization",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS giao diện chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .highlight-badge {
        background-color: #DC2626;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_geoclip_service(scope: str):
    """Nạp và cache GeoCLIPService để không phải tải lại model khi bấm nút"""
    return GeoCLIPService(root_dir=root_dir, scope=scope)

def main():
    st.markdown('<p class="main-title">🌍 GeoCLIP Vietnam: Visual Geo-localization Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Đồ án môn Phân tích và Trực quan hóa Dữ liệu — Trường Đại học Công nghệ Thông tin (UIT)</p>', unsafe_allow_html=True)
    st.divider()

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Cấu Hình Hệ Thống")
        
        scope_option = st.radio(
            "Chọn Phạm Vi Dữ Liệu:",
            (
                "🏛️ VN: Địa danh Biểu tượng (Iconic Landmarks)",
                "🗺️ VN: Mở rộng Toàn diện (26,353 POIs)",
                "🌐 Toàn cầu (Global 100K Mode)"
            ),
            index=0
        )
        
        if "Iconic" in scope_option:
            scope = "vietnam_iconic"
        elif "26,353" in scope_option:
            scope = "vietnam_all"
        else:
            scope = "global"
        
        top_k = st.slider("Số lượng vị trí dự đoán (Top-K):", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        st.subheader("🖼️ Hoặc chọn ảnh mẫu có sẵn:")
        sample_choice = st.selectbox(
            "Chọn ảnh mẫu:",
            ("Tải ảnh của bạn (Upload)", "Landmark 81 (TP.HCM)", "Kauai (Hawaii - USA)")
        )
        
        st.markdown("---")
        st.markdown("### 👥 Đội ngũ Phát triển (UIT):")
        st.caption("• **Tech Lead / AI Core**: Dev 1\n• **GIS Data & Metrics**: Dev 2\n• **Frontend & Map Viz**: Dev 3, 4, 5")

    # Nạp Service tương ứng
    with st.spinner("Đang nạp mô hình AI và Thư viện Tọa độ..."):
        service = load_geoclip_service(scope=scope)

    # --- TABS GIAO DIỆN ---
    tab1, tab2 = st.tabs(["🚀 Dự Đoán Vị Trí Ảnh", "📊 Khám Phá Dữ Liệu Địa Danh (Data Explorer)"])

    with tab1:
        col_left, col_right = st.columns([1, 1.3], gap="medium")

        input_image = None
        selected_sample_path = None

        with col_left:
            st.subheader("1. Tải lên Ảnh Cần Định Vị")
            
            if sample_choice == "Landmark 81 (TP.HCM)":
                selected_sample_path = os.path.join(root_dir, "data", "images.jpg")
                if os.path.exists(selected_sample_path):
                    input_image = Image.open(selected_sample_path).convert("RGB")
                    st.image(input_image, caption="Ảnh mẫu: Landmark 81 (TP. Hồ Chí Minh)", use_container_width=True)
            elif sample_choice == "Kauai (Hawaii - USA)":
                selected_sample_path = os.path.join(root_dir, "data", "sample_images", "Kauai.png")
                if os.path.exists(selected_sample_path):
                    input_image = Image.open(selected_sample_path).convert("RGB")
                    st.image(input_image, caption="Ảnh mẫu: Bãi biển Kauai (Hawaii)", use_container_width=True)
            else:
                uploaded_file = st.file_uploader("Kéo thả hoặc chọn file ảnh (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
                if uploaded_file is not None:
                    input_image = Image.open(uploaded_file).convert("RGB")
                    st.image(input_image, caption="Ảnh bạn vừa tải lên", use_container_width=True)

            predict_btn = st.button("🔍 BẮT ĐẦU ĐỊNH VỊ VỊ TRÍ", type="primary", use_container_width=True, disabled=(input_image is None))

        with col_right:
            st.subheader("2. Kết Quả Dự Đoán & Bản Đồ Không Gian")

            if predict_btn and input_image is not None:
                start_time = time.time()
                with st.spinner("Mô hình đang so khớp vector không gian..."):
                    predictions = service.predict(input_image, top_k=top_k)
                elapsed = time.time() - start_time

                st.success(f"⚡ Dự đoán hoàn tất trong **{elapsed:.4f} giây** trên **{len(service.gps_gallery):,} tọa độ**!")

                # Hiển thị thẻ Top 1 nổi bật
                top1 = predictions[0]
                st.markdown(f"""
                <div class="metric-card">
                    <span class="highlight-badge">⭐ TOP 1 DỰ ĐOÁN ({top1['prob_percent']}%)</span>
                    <h3 style="margin-top: 8px; color: #1E293B;">{top1.get('name', 'Địa điểm')}</h3>
                    <p style="margin: 2px 0; color: #475569;"><b>Tỉnh/Thành:</b> {top1.get('province', 'N/A')} | <b>Loại hình:</b> {top1.get('category', 'N/A')}</p>
                    <p style="margin: 2px 0; color: #64748B; font-size: 0.9rem;">{top1.get('description', '')}</p>
                    <p style="margin: 4px 0 8px 0; font-size: 0.85rem; color: #94A3B8;">Tọa độ: Lat {top1['lat']:.6f}, Lon {top1['lon']:.6f}</p>
                    <a href="{top1['gmaps_url']}" target="_blank" style="background-color: #10B981; color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 0.9rem; display: inline-block;">
                        📍 Mở trên Google Maps
                    </a>
                </div>
                """, unsafe_allow_html=True)

                # Hiển thị bản đồ tương tác Folium
                folium_map = create_prediction_map(predictions, zoom_start=13 if "vietnam" in scope else 3)
                components.html(folium_map._repr_html_(), height=420)

                # Bảng xếp hạng Top-K
                st.markdown("#### 📋 Bảng Xếp Hạng Top-K Dự Đoán")
                df_results = pd.DataFrame([
                    {
                        "Hạng": f"#{p['rank']}",
                        "Địa danh": p.get('name', 'N/A'),
                        "Tỉnh/Thành": p.get('province', 'N/A'),
                        "Loại hình": p.get('category', 'N/A'),
                        "Xác suất (%)": f"{p['prob_percent']:.2f}%",
                        "Vĩ độ (Lat)": f"{p['lat']:.4f}",
                        "Kinh độ (Lon)": f"{p['lon']:.4f}"
                    } for p in predictions
                ])
                st.dataframe(df_results, use_container_width=True, hide_index=True)
            else:
                st.info("👈 Vui lòng chọn hoặc tải ảnh lên ở cột bên trái và bấm **'Bắt đầu Định vị'** để xem bản đồ tương tác.")

    with tab2:
        st.subheader("📚 Cơ Sở Dữ Liệu Địa Danh Việt Nam")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            dataset_view = st.selectbox(
                "Xem tập dữ liệu:",
                ("🏛️ Địa danh Biểu tượng (Iconic Landmarks)", "🗺️ Mở rộng Toàn diện (26,353 POIs)")
            )
        
        target_csv = "vietnam_landmarks_iconic.csv" if "Iconic" in dataset_view else "vietnam_landmarks.csv"
        csv_path = os.path.join(root_dir, "data", target_csv)
        
        if os.path.exists(csv_path):
            df_display = pd.read_csv(csv_path)
            st.caption(f"Tổng số bản ghi: **{len(df_display):,} địa điểm**")
            
            # Bộ lọc theo thể loại
            categories = ["Tất cả"] + list(df_display['CATEGORY'].dropna().unique())
            with col_f2:
                selected_cat = st.selectbox("Lọc theo Loại hình:", categories)
                
            if selected_cat != "Tất cả":
                df_display = df_display[df_display['CATEGORY'] == selected_cat]
                
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning(f"Chưa tìm thấy file dữ liệu {target_csv}!")

if __name__ == "__main__":
    main()
