import streamlit as st
import cv2
import numpy as np
from detector import detect_license_plates
from ocr_reader import read_text_from_image
from PIL import Image

st.title("WebApp Nhận Diện Biển Số Xe")

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect biển số
    results = detect_license_plates(image)
    image_with_boxes = image.copy()

    if results:
        st.subheader("Ảnh và kết quả")
        for idx, (bbox, cropped) in enumerate(results):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"Plate {idx+1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            st.image(cropped, caption=f"Biển số {idx+1}", channels="BGR")
            text = read_text_from_image(cropped)
            st.success(f"Kết quả: {text}")

        # Hiển thị ảnh gốc có box
        st.image(image_with_boxes, caption="Ảnh có bounding box", channels="BGR")
    else:
        st.warning("Không phát hiện biển số nào.")
