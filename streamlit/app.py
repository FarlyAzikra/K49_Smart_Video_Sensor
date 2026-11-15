import os
import cv2
import time
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA BIRU MODERN
# ==========================================
st.set_page_config(
    page_title="Smart Video Sensor Dashboard - K49",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0D47A1 0%, #1976D2 100%);
        padding: 20px 25px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 { margin: 0; font-size: 28px; color: white; }
    .main-header p { margin: 5px 0 0 0; font-size: 14px; opacity: 0.9; }
    .metric-card {
        background-color: #161E2E;
        border: 1px solid #26334D;
        border-left: 5px solid #2196F3;
        padding: 18px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown("""
<div class="main-header">
    <h1>🎥 SMART VIDEO SENSOR DASHBOARD (K49)</h1>
    <p>Implementasi Object Detection, Object Tracking, Trajectory Analysis, & Vehicle Counting Berbasis Computer Vision</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI PEMUAT DATA CSV
# ==========================================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_counting = load_csv("results/counting_results.csv")
df_trajectory = load_csv("results/trajectory.csv")
df_tracking = load_csv("results/tracking_results.csv")
df_exp = load_csv("results/experiment_results.csv")

# ==========================================
# 3. SIDEBAR NAVIGASI (7 MENU WAJIB K49)
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/video-stabilization.png", width=64)
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "🏠 Home",
        "🎥 Video Tracking",
        "📊 Vehicle Statistics",
        "🗺️ Trajectory",
        "⚠️ Tracking Error",
        "📈 Performance",
        "⚖️ AI Ethics"
    ]
)

ANGGOTA_KELOMPOK = [
    {"nama": "Muhammad Farly Azikra", "nim": "2511500338", "peran": "Computer Vision & Detection"},
    {"nama": "Muhammad Zakariyya",    "nim": "2511500387", "peran": "Tracking & Analytics"},
    {"nama": "Ryan Babel Christian",  "nim": "2511501336", "peran": "Application & Research"},
]

st.sidebar.markdown("---")
st.sidebar.info("**Kelompok K49**\n\nModel: YOLO11s + ByteTrack\n\nTarget: Smart Video Sensor")

# ==========================================
# MENU 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.subheader("🏠 Konsep Dasar: Dari Kamera Biasa Menjadi Smart Video Sensor")
    st.markdown("### 👥 Anggota Kelompok K49")
    cols_team = st.columns(len(ANGGOTA_KELOMPOK))
    for idx, anggota in enumerate(ANGGOTA_KELOMPOK):
        with cols_team[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0 0 8px 0; color:#60A5FA; font-size:18px;">{anggota['nama']}</h4>
                <p style="margin:0; font-size:14px; color:#F8FAFC;"><b>NIM:</b> <code style="background:#0F172A; color:#38BDF8; padding:2px 6px; border-radius:4px;">{anggota['nim']}</code></p>
                <p style="margin:8px 0 0 0; font-size:13px; color:#94A3B8;"><i>Peran: {anggota['peran']}</i></p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.write(
        "Sistem ini mengembangkan kemampuan Computer Vision dari sekadar mendeteksi keberadaan objek "
        "menjadi sistem pelacakan pergerakan berkelanjutan antar-frame (*multi-object tracking*)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Tahap K47: Classification**\n\nMenjawab pertanyaan: *'Apa objek yang terlihat pada gambar?'*")
    with col2:
        st.info("**Tahap K48: Detection & Counting**\n\nMenjawab pertanyaan: *'Di mana posisi objek dan ada berapa jumlahnya?'*")
    with col3:
        st.success("**Tahap K49: Object Tracking (Aktif)**\n\nMenjawab pertanyaan: *'Objek yang sama bergerak ke arah mana?'*")

    st.markdown("### 🔄 Alur Kerja (*Pipeline*) Sistem")
    st.code(
        "CAMERA / VIDEO INPUT ➔ FRAME EXTRACTION ➔ OBJECT DETECTION (YOLO11s) ➔ "
        "OBJECT TRACKING (ByteTrack ID) ➔ CENTER POINT & TRAJECTORY ➔ "
        "LINE CROSSING & DIRECTION COUNTING ➔ VIDEO ANALYTICS",
        language="text"
    )

    st.markdown("### 🎯 Fitur Utama Dashboard")
    st.markdown("""
    - **Identitas Konsisten:** Mempertahankan nomor ID unik pada kendaraan yang sama dari frame ke frame.
    - **Analisis Lintasan (*Trajectory*):** Merekam koordinat titik tengah $(center\_x, center\_y)$ untuk memetakan jalur pergerakan.
    - **Penghitungan Dua Arah:** Menghitung jumlah kendaraan yang melintasi garis virtual secara terpisah ($A \\rightarrow B$ dan $B \\rightarrow A$) tanpa *double counting*.
    - **Evaluasi 10 Skenario:** Menguji ketahanan model pada variasi *confidence threshold*, kepadatan lalu lintas, kecepatan, hingga pencahayaan malam.
    """)

# ==========================================
# MENU 2: VIDEO TRACKING (UPLOAD & LIVE DEMO)
# ==========================================
elif menu == "🎥 Video Tracking":
    st.subheader("🎥 Video Object Tracking & Line Crossing")
    tab1, tab2 = st.tabs(["📤 Upload & Proses Video Baru", "🎬 Putar Hasil Eksperimen (E1–E10)"])

    with tab1:
        st.write("Unggah file video lalu lintas (`.mp4`) untuk memutar preview video asli dan menjalankan tracking.")
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            conf_val = st.slider("Confidence Threshold", 0.15, 0.85, 0.35, 0.05)
        with col_set2:
            line_pos = st.slider("Posisi Count Line (Rasio Tinggi Layar)", 0.15, 0.85, 0.40, 0.05)

        uploaded_file = st.file_uploader("Pilih Video (.mp4 / .avi / .mov)", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Simpan sementara file yang diunggah
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.getvalue())
            tfile.flush()

            # TAMPILKAN PREVIEW VIDEO ASLI & SIMULASI GARIS SEBELUM DIPROSES
            col_vid1, col_vid2 = st.columns(2)
            with col_vid1:
                st.markdown("**🎬 Preview Video Asli (Input)**")
                st.video(uploaded_file.getvalue())

            with col_vid2:
                st.markdown(f"**📏 Simulasi Posisi Count Line ({line_pos:.2f})**")
                cap_prev = cv2.VideoCapture(tfile.name)
                ret_prev, frame_prev = cap_prev.read()
                cap_prev.release()
                if ret_prev:
                    h_p, w_p, _ = frame_prev.shape
                    ly_p = int(h_p * line_pos)
                    cv2.line(frame_prev, (0, ly_p), (w_p, ly_p), (0, 255, 255), 4)
                    cv2.putText(frame_prev, f"COUNT LINE ({line_pos:.2f})", (15, ly_p - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    st.image(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("---")

            if st.button("▶️ Mulai Proses Tracking", type="primary"):
                with st.spinner("Sedang memproses video frame-demi-frame... Mohon tunggu."):
                    model_path = "model/detection_model.pt" if os.path.exists("model/detection_model.pt") else "yolo11s.pt"
                    model = YOLO(model_path)
                    vehicle_classes = {1: "motorcycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

                    cap = cv2.VideoCapture(tfile.name)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    line_y = int(height * line_pos)

                    raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

                    track_history = defaultdict(list)
                    counted_ids = set()
                    c_ab, c_ba = 0, 0
                    frames = 0
                    t0 = time.time()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames += 1
                        results = model.track(frame, persist=True, conf=conf_val, imgsz=960,
                                              classes=list(vehicle_classes.keys()), tracker="bytetrack.yaml", verbose=False)

                        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            tids = results[0].boxes.id.int().cpu().tolist()
                            cids = results[0].boxes.cls.int().cpu().tolist()

                            for box, tid, cid in zip(boxes, tids, cids):
                                x1, y1, x2, y2 = map(int, box)
                                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                track_history[tid].append((cx, cy))
                                label = vehicle_classes[cid]

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 2)
                                cv2.putText(frame, f"ID {tid} | {label}", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                pts = track_history[tid][-20:]
                                for i in range(1, len(pts)):
                                    cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2)

                                if len(track_history[tid]) >= 2 and tid not in counted_ids:
                                    py = track_history[tid][-2][1]
                                    if py < line_y <= cy:
                                        c_ab += 1
                                        counted_ids.add(tid)
                                    elif py > line_y >= cy:
                                        c_ba += 1
                                        counted_ids.add(tid)

                        cv2.rectangle(frame, (10, 10), (420, 48), (0, 0, 0), -1)
                        cv2.putText(frame, f"Total: {c_ab+c_ba} (A->B:{c_ab}, B->A:{c_ba})",
                                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                        out.write(frame)

                    cap.release()
                    out.release()
                    fps_proc = round(frames / (time.time() - t0), 2)

                    # Konversi video hasil ke H.264 agar bisa diputar langsung di browser web
                    web_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    try:
                        import subprocess, imageio_ffmpeg
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                        subprocess.run(
                            [ffmpeg_exe, "-y", "-i", raw_out, "-vcodec", "libx264", "-pix_fmt", "yuv420p", web_out],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )
                        final_video_path = web_out
                    except Exception:
                        final_video_path = raw_out

                    st.session_state["last_video_path"] = final_video_path
                    st.session_state["last_stats"] = (c_ab, c_ba, fps_proc)

            # Tampilkan hasil jika sudah selesai diproses
            if "last_video_path" in st.session_state and os.path.exists(st.session_state["last_video_path"]):
                c_ab, c_ba, fps_proc = st.session_state["last_stats"]
                st.success(f"✅ Selesai! Total Terhitung: {c_ab + c_ba} kendaraan (A➔B: {c_ab}, B➔A: {c_ba}) | Kecepatan: {fps_proc} FPS")
                st.markdown("**🎥 Hasil Video Tracking**")
                st.video(st.session_state["last_video_path"])
                with open(st.session_state["last_video_path"], "rb") as f_dl:
                    st.download_button("⬇️ Download Video Hasil Tracking", f_dl, file_name="hasil_tracking_k49.mp4")

    with tab2:
        exp_dir = "results/experiment_videos"
        if os.path.exists(exp_dir) and len(os.listdir(exp_dir)) > 0:
            vid_files = sorted(os.listdir(exp_dir))
            selected_vid = st.selectbox("Pilih Video Hasil Eksperimen:", vid_files)
            vid_path = os.path.join(exp_dir, selected_vid)
            st.video(vid_path)
            with open(vid_path, "rb") as f:
                st.download_button(f"⬇️ Download {selected_vid}", f, file_name=selected_vid)
        else:
            st.warning("Folder `results/experiment_videos` masih kosong. Jalankan `run_experiments.py` terlebih dahulu.")

# ==========================================
# MENU 3: VEHICLE STATISTICS
# ==========================================
elif menu == "📊 Vehicle Statistics":
    st.subheader("📊 Statistik Kendaraan & Distribusi Arah (Baseline E1)")
    if df_counting is not None:
        row = df_counting.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Kendaraan", int(row["total_vehicles"]))
        c2.metric("Arah A ➔ B (Turun)", int(row["direction_A_to_B"]))
        c3.metric("Arah B ➔ A (Naik)", int(row["direction_B_to_A"]))
        dominan = "A ➔ B" if row["direction_A_to_B"] >= row["direction_B_to_A"] else "B ➔ A"
        c4.metric("Arus Dominan", dominan)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            df_jenis = pd.DataFrame({
                "Jenis Kendaraan": ["Car", "Motorcycle", "Bus", "Truck"],
                "Jumlah": [row["car"], row["motorcycle"], row["bus"], row["truck"]]
            })
            fig_bar = px.bar(df_jenis, x="Jenis Kendaraan", y="Jumlah", color="Jenis Kendaraan",
                             title="Jumlah Kendaraan Berdasarkan Jenis", text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            df_arah = pd.DataFrame({
                "Arah Pergerakan": ["Arah A ➔ B", "Arah B ➔ A"],
                "Jumlah": [row["direction_A_to_B"], row["direction_B_to_A"]]
            })
            fig_pie = px.pie(df_arah, names="Arah Pergerakan", values="Jumlah", hole=0.4,
                             title="Proporsi Arah Pergerakan Kendaraan")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Data `results/counting_results.csv` belum ditemukan.")

# ==========================================
# MENU 4: TRAJECTORY
# ==========================================
elif menu == "🗺️ Trajectory":
    st.subheader("🗺️ Analisis Lintasan Pergerakan Objek (Trajectory)")
    if df_trajectory is not None and not df_trajectory.empty:
        unique_ids = sorted(df_trajectory["track_id"].unique())
        st.write(f"Total objek unik yang terekam lintasannya: **{len(unique_ids)} ID**")

        selected_ids = st.multiselect(
            "Filter ID Kendaraan yang Ingin Ditampilkan (kosongkan untuk tampilkan 15 ID pertama):",
            options=unique_ids,
            default=unique_ids[:15]
        )

        df_plot = df_trajectory[df_trajectory["track_id"].isin(selected_ids)] if selected_ids else df_trajectory[df_trajectory["track_id"].isin(unique_ids[:15])]
        df_plot = df_plot.copy()
        df_plot["track_id"] = df_plot["track_id"].astype(str)

        fig_traj = px.line(
            df_plot, x="center_x", y="center_y", color="track_id", markers=True,
            title="Peta Koordinat Lintasan Objek (Center Point Trajectory)",
            labels={"center_x": "Koordinat X (Piksel Horizontal)", "center_y": "Koordinat Y (Piksel Vertikal)", "track_id": "ID Objek"}
        )
        # Balik sumbu Y agar sesuai dengan koordinat gambar/video (0,0 di pojok kiri atas)
        fig_traj.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_traj, use_container_width=True)

        with st.expander("📄 Lihat Data Mentah Koordinat Trajectory (`trajectory.csv`)"):
            st.dataframe(df_plot.head(100), use_container_width=True)
    else:
        st.warning("Data `results/trajectory.csv` belum ditemukan.")

# ==========================================
# MENU 5: TRACKING ERROR
# ==========================================
elif menu == "⚠️ Tracking Error":
    st.subheader("⚠️ Analisis Kesalahan Deteksi & Tracking (Error Analysis)")
    st.write("Dokumentasi kelemahan sistem pada berbagai kondisi lapangan untuk menjawab RQ1–RQ4.")

    err_data = pd.DataFrame([
        {"Jenis Error": "Missed Object", "Penyebab Utama": "Ukuran motor terlalu kecil pada sudut kamera tinggi & confidence threshold tinggi (0.70).", "Dampak": "Kendaraan melintas tanpa terhitung (Under-counting)."},
        {"Jenis Error": "Misclassification", "Penyebab Utama": "Sudut pandang belakang/atas membuat SUV atau sedan besar mirip bentuk bus/truk.", "Dampak": "Distribusi jenis kendaraan bergeser."},
        {"Jenis Error": "ID Switch", "Penyebab Utama": "Kendaraan saling berdekatan atau tertutup sementara (occlusion) saat menyalip.", "Dampak": "1 kendaraan mendapat nomor ID baru di tengah jalan."},
        {"Jenis Error": "ROI / Count Line Miss", "Penyebab Utama": "Garis penghitung terlalu rendah (0.60) sehingga jalur kiri keluar frame sebelum menyentuh garis.", "Dampak": "Arah berlawanan tidak terhitung (0)."}
    ])
    st.table(err_data)

    st.markdown("### 📸 Bukti Visual Temuan Error di Lapangan")
    sample_dir = "results/error_samples"
    if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
        img_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        cols = st.columns(min(len(img_files), 3))
        for idx, img_name in enumerate(img_files):
            with cols[idx % 3]:
                st.image(os.path.join(sample_dir, img_name), caption=img_name, use_container_width=True)
    else:
        st.info("Tip: Simpan screenshot hasil tracking yang menunjukkan error ke folder `results/error_samples/` agar otomatis muncul di sini.")

# ==========================================
# MENU 6: PERFORMANCE
# ==========================================
elif menu == "📈 Performance":
    st.subheader("📈 Evaluasi Performa & Hasil 10 Eksperimen (E1–E10)")
    if df_exp is not None and not df_exp.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rata-rata Kecepatan", f"{df_exp['Avg_FPS'].mean():.2f} FPS")
        c2.metric("Rata-rata Inference Time", f"{df_exp['Inference_Time_ms'].mean():.2f} ms/frame")
        c3.metric("Total Skenario Diuji", f"{len(df_exp)} Eksperimen")

        st.markdown("### 📋 Tabel Rekapitulasi Eksperimen (`experiment_results.csv`)")
        st.dataframe(df_exp, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_fps = px.bar(df_exp, x="Experiment_ID", y="Avg_FPS", color="Skenario",
                             title="Perbandingan Kecepatan Pemrosesan (FPS) per Skenario", text_auto=True)
            st.plotly_chart(fig_fps, use_container_width=True)

        with col_p2:
            fig_cnt = px.bar(df_exp, x="Experiment_ID", y=["Total_Counted_AI", "Total_Unique_IDs"],
                             barmode="group", title="Perbandingan Kendaraan Terhitung vs Total ID Unik Terbentuk")
            st.plotly_chart(fig_cnt, use_container_width=True)
    else:
        st.warning("Data `results/experiment_results.csv` belum ditemukan.")

# ==========================================
# MENU 7: AI ETHICS
# ==========================================
elif menu == "⚖️ AI Ethics":
    st.subheader("⚖️ Etika Kecerdasan Buatan (AI Ethics) & 9 Nilai Kebudiluhuran")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("### 🛡️ Prinsip AI Ethics pada Smart Video Sensor")
        st.markdown("""
        1. **Privacy (Privasi Visual):** Rekaman jalan raya dapat menangkap wajah pengendara maupun nomor polisi kendaraan. Sistem dibatasi hanya memproses bentuk kendaraan (*bounding box*) tanpa melakukan pengenalan wajah (*face recognition*) atau pembacaan plat nomor (*ANPR*).
        2. **Data Protection:** Dataset video uji disimpan secara lokal dan hanya digunakan untuk keperluan akademik.
        3. **Anti-Surveillance:** Teknologi *object tracking* dirancang untuk analisis kepadatan lalu lintas, bukan pengawasan individu tanpa dasar hukum.
        4. **Accuracy & Limitations:** Model AI memiliki keterbatasan (seperti *ID switch* dan *missed object* pada malam hari) sehingga tidak boleh dijadikan satu-satunya dasar pengambilan keputusan krusial.
        5. **Human Oversight & Transparency:** Operator manusia tetap memegang kendali penuh atas interpretasi data yang dihasilkan dashboard.
        """)

    with col_e2:
        st.markdown("### 🌟 Implementasi 9 Nilai Kebudiluhuran")
        nilai_df = pd.DataFrame([
            {"Nilai": "Sabar Mensyukuri", "Implementasi dalam Proyek K49": "Memanfaatkan teknologi AI secara bijak untuk kemaslahatan analisis lalu lintas."},
            {"Nilai": "Cinta Kasih", "Implementasi dalam Proyek K49": "Mengutamakan keselamatan pengguna jalan dan perlindungan privasi individu."},
            {"Nilai": "Rendah Hati", "Implementasi dalam Proyek K49": "Mengakui secara terbuka keterbatasan akurasi model deteksi pada kondisi ekstrem."},
            {"Nilai": "Suka Menolong", "Implementasi dalam Proyek K49": "Menyajikan visualisasi data lalu lintas yang mudah dipahami pengambil kebijakan."},
            {"Nilai": "Kerja Sama", "Implementasi dalam Proyek K49": "Kolaborasi tim dalam pembagian tugas Computer Vision, Tracking Analytics, dan Riset."},
            {"Nilai": "Jujur", "Implementasi dalam Proyek K49": "Melaporkan seluruh error (ID switch, missed detection) apa adanya tanpa manipulasi data."},
            {"Nilai": "Tanggung Jawab", "Implementasi dalam Proyek K49": "Menjaga keamanan file video dataset agar tidak disalahgunakan."},
            {"Nilai": "Toleransi", "Implementasi dalam Proyek K49": "Menghargai perbedaan metode dan parameter tracker dalam diskusi penelitian."},
            {"Nilai": "Sopan Santun", "Implementasi dalam Proyek K49": "Menjaga etika penggunaan data visual ruang publik."}
        ])
        st.dataframe(nilai_df, use_container_width=True, hide_index=True)