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
VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/x-msvideo", "video/quicktime"}

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
                x1, y1, x2, y2 = box.int().tolist()
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

# ================== DETECTION PAGE ==================
elif nav == "Deteksi":
    left, right = st.columns([3, 1])
    with right:
        st.subheader("⚙️ Kontrol")
        source_type = st.radio("Sumber Input", ["📷 Gambar", "🎥 Video", "📹 Webcam"], index=0)
        conf_th = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        iou_th = st.slider("IoU NMS", 0.1, 0.9, 0.45, 0.05)
        st.checkbox("Bunyikan Alarm saat Terdeteksi", value=st.session_state.alarm_enabled, key="alarm_enabled")
        st.markdown("---")
        if source_type in ("📷 Gambar", "🎥 Video"):
            uploaded_file = st.file_uploader("Upload file", type=["jpg","jpeg","png","webp","bmp","mp4","avi","mov"], accept_multiple_files=False)
        else:
            uploaded_file = None
        st.caption("Tips: Untuk video panjang, pertimbangkan resolusi kecil agar FPS stabil.")

    with left:
        st.subheader("👁️‍🗨️ Tampilan Deteksi")
        # ========== IMAGE ==========
        if source_type == "📷 Gambar":
            if uploaded_file is None:
                st.info("Unggah gambar untuk diproses.")
            elif not is_image_file(uploaded_file):
                st.error("File yang diunggah bukan gambar. Pilih file JPEG/PNG/WebP/BMP.")
            else:
                try:
                    image = np.array(Image.open(uploaded_file).convert('RGB'))
                except UnidentifiedImageError:
                    st.error("Gagal membaca gambar. Coba format lain.")
                else:
                    # Inference (gunakan conf & iou agar lebih hemat pasca-filter)
                    results = model(image, conf=conf_th, iou=iou_th)
                    frames, total_fire, avg_conf = draw_fire_only(results, min_conf=conf_th)
                    st.image(frames[0], caption="Hasil Deteksi Api", use_column_width=True)

                    # Metrics (cards di samping sudah ada; tampilkan ringkas di bawah untuk gambar)
                    m1, m2 = st.columns(2)
                    m1.metric("Jumlah Api Terdeteksi", total_fire)
                    m2.metric("Rata-rata Confidence", f"{avg_conf*100:.1f}%")

                    # Log
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Image",
                        "detections": int(total_fire),
                        "avg_conf": float(avg_conf),
                        "note": uploaded_file.name,
                    })

                    if total_fire > 0:
                        st.error("⚠️ Kebakaran terdeteksi!")
                        play_alarm(loop=True)
                    else:
                        st.success("✅ Tidak ada api yang terdeteksi")

        # ========== VIDEO ==========
        elif source_type == "🎥 Video":
            if uploaded_file is None:
                st.info("Unggah video untuk diproses.")
            elif not is_video_file(uploaded_file):
                st.error("File yang diunggah bukan video.")
            else:
                # Simpan sementara & buka dengan OpenCV
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)

                frame_view = st.empty()
                det_count, conf_acc = 0, []
                frame_skip = st.slider("Lewati tiap N frame (untuk hemat CPU)", 1, 10, 1)
                run = st.checkbox("Mulai proses video", value=False)

                f_idx = 0
                while run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    f_idx += 1
                    if f_idx % frame_skip != 0:
                        continue

                    results = model(frame, conf=conf_th, iou=iou_th)
                    frames, total_fire, avg_conf = draw_fire_only(results, min_conf=conf_th)
                    frame_view.image(frames[0], channels="BGR", use_column_width=True)

                    if total_fire > 0:
                        det_count += total_fire
                        if avg_conf > 0:
                            conf_acc.append(avg_conf)
                        play_alarm(loop=True)

                cap.release()

                # Log setelah selesai/berhenti
                if f_idx > 0:
                    avg_conf_overall = float(np.mean(conf_acc)) if conf_acc else 0.0
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Video",
                        "detections": int(det_count),
                        "avg_conf": avg_conf_overall,
                        "note": uploaded_file.name,
                    })
                    st.success("Proses video selesai / dihentikan.")

        # ========== WEBCAM ==========
        elif source_type == "📹 Webcam":
            device_index = st.number_input("Index Kamera", min_value=0, value=0, step=1)
            cap = cv2.VideoCapture(int(device_index))
            if not cap.isOpened():
                st.error("Kamera tidak dapat dibuka. Coba index lain.")
            else:
                frame_view = st.empty()
                run_cam = st.checkbox("Mulai Webcam", value=False)
                det_total, conf_acc = 0, []

                while run_cam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Frame tidak terbaca.")
                        break

                    results = model(frame, conf=conf_th, iou=iou_th)
                    frames, total_fire, avg_conf = draw_fire_only(results, min_conf=conf_th)
                    frame_view.image(frames[0], channels="BGR", use_column_width=True)

                    if total_fire > 0:
                        det_total += total_fire
                        if avg_conf > 0:
                            conf_acc.append(avg_conf)
                        play_alarm(loop=True)

                cap.release()

                if det_total >= 0:
                    avg_conf_overall = float(np.mean(conf_acc)) if conf_acc else 0.0
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Webcam",
                        "detections": int(det_total),
                        "avg_conf": avg_conf_overall,
                        "note": f"device_index={device_index}",
                    })
                    st.info("Webcam dihentikan.")

# ================== HISTORY / EXPORT PAGE ==================
elif nav == "Riwayat & Ekspor":
    st.subheader("🗂️ Riwayat Deteksi")
    if not st.session_state.logs:
        st.info("Belum ada data.")
    else:
        # Tabel sederhana
        st.dataframe(st.session_state.logs, use_container_width=True)
        # Download CSV
        csv_bytes = logs_to_csv_bytes()
        st.download_button("⬇️ Unduh CSV Riwayat", data=csv_bytes, file_name="fire_detection_logs.csv", mime="text/csv")

# ================== SETTINGS ==================
elif nav == "Pengaturan":
    st.subheader("🔧 Pengaturan")
    st.toggle("Aktifkan bunyi alarm", key="alarm_enabled")
    st.caption("Catatan: beberapa browser memerlukan interaksi pengguna terlebih dahulu agar audio dapat autoplay.")
    st.markdown("---")
    st.write("Hapus riwayat deteksi:")
    if st.button("🧹 Bersihkan Riwayat"):
        st.session_state.logs = []
        st.success("Riwayat dihapus.")
