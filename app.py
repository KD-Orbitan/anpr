"""Run with: python -m streamlit run app.py"""
import cv2
import numpy as np
import streamlit as st

from plateocr.inference import Pipeline, Recognizer
from plateocr.project import ROOT, read_json

st.set_page_config(page_title="Nhận diện biển số", layout="centered")
st.title("Nhận diện biển số xe")
model = st.selectbox("Model OCR", list(read_json(ROOT / "models/registry.json")["models"]))
mode = st.radio("Loại ảnh", ["Ảnh xe", "Ảnh biển số đã cắt"], horizontal=True)


@st.cache_resource
def load_model(name, detect):
    return Pipeline(name) if detect else Recognizer(name)


uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
if uploaded:
    image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không đọc được ảnh.")
    else:
        try:
            engine = load_model(model, mode == "Ảnh xe")
            # Paddle predictors have mutable buffers; serialize shared Streamlit calls.
            import threading

            @st.cache_resource
            def prediction_lock():
                return threading.Lock()

            with prediction_lock():
                result = engine.predict(image)
            if mode == "Ảnh xe":
                for plate in result:
                    x1, y1, x2, y2 = plate["box"]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, plate["text"], (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not result:
                    st.info("Không tìm thấy biển số.")
            st.image(image, channels="BGR")
            st.json(result)
            st.caption("Confidence là điểm tin cậy của dự đoán, không phải accuracy của model.")
        except Exception as error:
            st.error(str(error))
