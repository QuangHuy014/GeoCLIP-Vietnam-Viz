# ==============================================================================
# MODULE: Map Builder & Visualization (Frontend UI Team: Dev 3)
# RESPONSIBILITY: Generate Folium / Leaflet interactive map with custom markers
# ==============================================================================

import folium
from typing import List, Dict, Tuple

def create_prediction_map(predictions: List[Dict],
                          map_center: List[float] = None,
                          zoom_start: int = 6,
                          ground_truth: Tuple[float, float] = None,
                          save_html_path: str = None) -> folium.Map:
    """
    Tạo bản đồ tương tác Folium hiển thị Top-K kết quả dự đoán địa danh Việt Nam.
    Args:
        predictions: Danh sách kết quả từ GeoCLIPService.predict()
        map_center: [Lat, Lon] trung tâm bản đồ (Mặc định: Đà Nẵng [16.047, 108.206])
        zoom_start: Mức phóng to ban đầu (mặc định = 6 để thấy toàn cảnh Việt Nam)
        ground_truth: (Lat, Lon) tọa độ thực tế nếu có để đo khoảng cách và vẽ đường nối
        save_html_path: Đường dẫn lưu file HTML nếu muốn xuất file
    Returns:
        Đối tượng folium.Map
    """
    if map_center is None:
        if ground_truth:
            map_center = [ground_truth[0], ground_truth[1]]
            zoom_start = 12
        elif predictions and len(predictions) > 0:
            map_center = [predictions[0]['lat'], predictions[0]['lon']]
            zoom_start = 12
        else:
            map_center = [16.047079, 108.206230]
            zoom_start = 6

    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Nếu có tọa độ thực tế (Ground Truth), vẽ điểm xanh lá cây
    if ground_truth:
        gt_lat, gt_lon = ground_truth
        folium.Marker(
            location=[gt_lat, gt_lon],
            popup=folium.Popup(f"<b>🎯 VỊ TRÍ THỰC TẾ (Ground Truth)</b><br>Lat: {gt_lat:.6f}, Lon: {gt_lon:.6f}", max_width=250),
            tooltip="🎯 Vị trí thực tế",
            icon=folium.Icon(color='green', icon='ok-sign', prefix='glyphicon')
        ).add_to(m)

        # Vẽ đường Geodesic nối từ Tọa độ thật -> Top 1 Dự đoán
        if predictions and len(predictions) > 0:
            pred_lat, pred_lon = predictions[0]['lat'], predictions[0]['lon']
            folium.PolyLine(
                locations=[[gt_lat, gt_lon], [pred_lat, pred_lon]],
                color="#E53E3E",
                weight=3,
                dash_array="8, 8",
                tooltip="Đường sai số Geodesic (Khoảng cách đường chim bay)"
            ).add_to(m)

    # Thêm các điểm ghim dự đoán (Markers)
    for p in predictions:
        rank = p.get('rank', 1)
        lat = p['lat']
        lon = p['lon']
        name = p.get('name', 'Địa điểm')
        category = p.get('category', 'Danh lam')
        province = p.get('province', 'Việt Nam')
        prob = p.get('prob_percent', 0.0)
        gmaps_url = p.get('gmaps_url', f"https://www.google.com/maps?q={lat},{lon}")

        # Định dạng Popup HTML chuyên nghiệp
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 220px; font-size: 13px;">
            <div style="background-color: {'#E53E3E' if rank == 1 else '#3182CE'}; color: white; padding: 6px; border-radius: 4px; font-weight: bold; text-align: center;">
                {'⭐ HẠNG 1' if rank == 1 else f'HẠNG #{rank}'} ({prob:.2f}%)
            </div>
            <div style="padding: 8px 4px;">
                <h4 style="margin: 4px 0; color: #2D3748;">{name}</h4>
                <p style="margin: 2px 0; color: #718096;"><b>Tỉnh/Thành:</b> {province}</p>
                <p style="margin: 2px 0; color: #718096;"><b>Loại hình:</b> {category}</p>
                <p style="margin: 2px 0; font-size: 11px; color: #A0AEC0;">Tọa độ: {lat:.4f}, {lon:.4f}</p>
                <div style="margin-top: 8px; text-align: center;">
                    <a href="{gmaps_url}" target="_blank" style="background-color: #38A169; color: white; padding: 4px 10px; border-radius: 4px; text-decoration: none; font-size: 12px; display: inline-block;">
                        📍 Xem trên Google Maps
                    </a>
                </div>
            </div>
        </div>
        """

        # Chọn màu sắc và biểu tượng theo thứ hạng
        if rank == 1:
            icon_color = 'red'
            icon_name = 'star'
        elif rank == 2:
            icon_color = 'orange'
            icon_name = 'info-sign'
        else:
            icon_color = 'blue'
            icon_name = 'map-marker'

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Hạng #{rank}: {name} ({prob:.2f}%)",
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='glyphicon')
        ).add_to(m)

        # Thêm vòng tròn bán kính ảnh hưởng cho Top-1 (25km cấp thành phố)
        if rank == 1:
            folium.Circle(
                location=[lat, lon],
                radius=15000,  # 15km
                color="#E53E3E",
                weight=1.5,
                fill=True,
                fill_color="#FEB2B2",
                fill_opacity=0.3,
                popup="Vùng xác suất tập trung (Bán kính 15km)"
            ).add_to(m)

    if save_html_path:
        m.save(save_html_path)
        print(f"[MapBuilder] Saved interactive map to: {save_html_path}")

    return m
