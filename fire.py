from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Ganti path ini sesuai lokasi modelmu
MODEL_PATH = "best.pt"

# Load model hasil training deteksi api
model = YOLO(MODEL_PATH)

# Ganti class ID sesuai hasil training
FIRE_CLASS_ID = 0  # misal: 0 = Fire

# Fungsi untuk menggambar hanya deteksi "Fire"
def draw_fire_only(results, class_id=FIRE_CLASS_ID, min_conf=0.5):
    """
    Mengembalikan frame dengan hanya deteksi Fire + confidence score
    """
    annotated_frames = []
    for r in results:
        frame = r.orig_img.copy()
        boxes = r.boxes
        if boxes is None:
            annotated_frames.append(frame)
            continue

        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            if int(cls.item()) == class_id and conf.item() >= min_conf:
                x1, y1, x2, y2 = box.int().tolist()
                confidence = f"{conf.item()*100:.1f}%"
                label = f"Fire {confidence}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        annotated_frames.append(frame)
    return annotated_frames

# Judul halaman
st.title("🔥 Deteksi Api Otomatis dengan YOLOv11 + Streamlit")

# Upload file (gambar/video)
uploaded_file = st.file_uploader("Upload gambar atau video", 
                                 type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# ==== Gambar ====
if uploaded_file is not None and uploaded_file.type.startswith('image'):
    image = np.array(Image.open(uploaded_file))
    results = model(image)
    frames = draw_fire_only(results)
    st.image(frames[0], caption="Deteksi Api", use_container_width=True)

# ==== Video ====
elif uploaded_file is not None and uploaded_file.type.startswith('video'):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frames = draw_fire_only(results)
        stframe.image(frames[0], channels="BGR")
    cap.release()

# ==== Webcam ====
elif uploaded_file is None:
    st.info("Gunakan webcam untuk deteksi api secara real-time")
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            frames = draw_fire_only(results, min_conf=0.5)
            stframe.image(frames[0], channels="BGR")
    cap.release()
