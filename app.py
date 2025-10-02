from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image, UnidentifiedImageError
import base64
import io
import csv
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
MODEL_PATH = "best1.pt"
FIRE_CLASS_ID = 0
ALARM_FILE = "alarm.mp3"  # pastikan file ini ada di folder proyek

# ================== CACHED RESOURCES ==================
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ================== SESSION STATE ==================
if "logs" not in st.session_state:
    st.session_state.logs = []  # list of dicts: time, source, detections, avg_conf
if "alarm_enabled" not in st.session_state:
    st.session_state.alarm_enabled = True

# ================== HELPERS ==================
IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}

def is_image_file(upload):
    return upload is not None and upload.type in IMAGE_TYPES

def is_video_file(upload):
    return upload is not None and upload.type.split("/")[0] == "video"

# Draw only Fire class & collect stats
def draw_fire_only(results, class_id=FIRE_CLASS_ID, min_conf=0.5):
    annotated_frames = []
    total_fire, confs = 0, []
    
    for r in results:
        frame = r.orig_img.copy()
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            annotated_frames.append(frame)
            continue

        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            c = float(conf.item()) if hasattr(conf, 'item') else float(conf)
            if int(cls.item()) == class_id and c >= min_conf:
                # FIX: .int().tolist() diganti biar aman di Python 3.10
                x1, y1, x2, y2 = box.cpu().numpy().astype(int).tolist()
                conf_pct = c * 100
                label = f"🔥 Fire {conf_pct:.1f}%"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 20, 60), 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 20, 60), 2)
                total_fire += 1
                confs.append(c)
        
        annotated_frames.append(frame)

    avg_conf = float(np.mean(confs)) if confs else 0.0
    return annotated_frames, total_fire, avg_conf

# HTML5 audio (autoplay may require a prior user interaction in some browsers)
def play_alarm(loop=True):
    if not st.session_state.alarm_enabled:
        return
    try:
        with open(ALARM_FILE, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        loop_attr = " loop" if loop else ""
        st.markdown(
            f"""
            <audio autoplay{loop_attr}>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("File alarm.mp3 tidak ditemukan. Letakkan di folder proyek.")

# Export logs to CSV bytes
def logs_to_csv_bytes():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp","source","detections","avg_conf","note"])
    writer.writeheader()
    for row in st.session_state.logs:
        writer.writerow(row)
    return output.getvalue().encode()

# ================== PAGE CONFIG & SIDEBAR ==================
st.set_page_config(page_title="Fire Detection Dashboard", page_icon="🔥", layout="wide")

with st.sidebar:
    st.title("🔥 Fire Detection")
    nav = st.radio("Navigasi", ["Dashboard", "Deteksi", "Riwayat & Ekspor", "Pengaturan"], index=0)
    st.markdown("---")
    st.caption("v1.0 • YOLOv11 + Streamlit")

# ================== HEADER ==================
st.markdown(
    """
    <div style="text-align:center; padding: 8px 0 16px 0;">
        <h1 style="margin-bottom:6px;color:#d63031;">Fire Detection Dashboard</h1>
        <p style="color:#636e72;">Deteksi kebakaran otomatis dengan <b>YOLOv11</b>. Unggah gambar/video atau gunakan webcam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================== DASHBOARD PAGE ==================
if nav == "Dashboard":
    c1, c2, c3 = st.columns(3)
    with c1:
        total_events = sum(1 for r in st.session_state.logs if int(r.get("detections",0)) > 0)
        st.metric("Total Peristiwa Terdeteksi", total_events)
    with c2:
        last = st.session_state.logs[-1]["timestamp"] if st.session_state.logs else "-"
        st.metric("Terakhir Diproses", last)
    with c3:
        status = "Aktif" if st.session_state.alarm_enabled else "Senyap"
        st.metric("Status Alarm", status)

    st.markdown("---")
    st.subheader("Ringkasan Terbaru")
    recent = st.session_state.logs[-10:][::-1]
    if recent:
        for r in recent:
            st.write(f"[{r['timestamp']}] • Source: {r['source']} • Detections: {r['detections']} • Avg conf: {float(r['avg_conf']):.2f}")
    else:
        st.info("Belum ada riwayat. Jalankan deteksi di menu \"Deteksi\".")

# ================== HISTORY / EXPORT PAGE ==================
elif nav == "Riwayat & Ekspor":
    st.subheader("🗂️ Riwayat Deteksi")
    if not st.session_state.logs:
        st.info("Belum ada data.")
    else:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df, use_container_width=True)
        csv_bytes = logs_to_csv_bytes()
        st.download_button("⬇️ Unduh CSV Riwayat", data=csv_bytes, file_name="fire_detection_logs.csv", mime="text/csv")

# ================== SETTINGS ==================
elif nav == "Pengaturan":
    st.subheader("🔧 Pengaturan")
    # FIX: toggle hanya ada di Streamlit >=1.29
    if hasattr(st, "toggle"):
        st.toggle("Aktifkan bunyi alarm", key="alarm_enabled")
    else:
        st.checkbox("Aktifkan bunyi alarm", key="alarm_enabled")
    st.caption("Catatan: beberapa browser memerlukan interaksi pengguna terlebih dahulu agar audio dapat autoplay.")
    st.markdown("---")
    st.write("Hapus riwayat deteksi:")
    if st.button("🧹 Bersihkan Riwayat"):
        st.session_state.logs = []
        st.success("Riwayat dihapus.")
