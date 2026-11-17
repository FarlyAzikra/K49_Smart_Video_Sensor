import os
import cv2
import time
import torch
import shutil
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"🖥️ Perangkat Komputasi Aktif: {device_name} (device={DEVICE})")

folders = ["dataset/videos", "model", "tracking", "results/experiment_videos", "streamlit"]
for f in folders:
    os.makedirs(f, exist_ok=True)

VIDEO_DIR = "dataset/videos"
OUTPUT_VIDEO_DIR = "results/experiment_videos"

experiments = [
    {"id": "E1",  "skenario": "Kondisi Normal (Baseline)",   "video": "video_normal.mp4",     "conf": 0.35, "line_ratio": 0.25, "tracker": "bytetrack.yaml"},
    {"id": "E2",  "skenario": "Confidence Threshold 0.30",   "video": "video_normal.mp4",     "conf": 0.30, "line_ratio": 0.25, "tracker": "bytetrack.yaml"},
    {"id": "E3",  "skenario": "Confidence Threshold 0.50",   "video": "video_normal.mp4",     "conf": 0.50, "line_ratio": 0.25, "tracker": "bytetrack.yaml"},
    {"id": "E4",  "skenario": "Confidence Threshold 0.70",   "video": "video_normal.mp4",     "conf": 0.70, "line_ratio": 0.25, "tracker": "bytetrack.yaml"},
    {"id": "E5",  "skenario": "Sedikit Kendaraan (Sepi)",    "video": "video_sepi.mp4",       "conf": 0.35, "line_ratio": 0.40, "tracker": "bytetrack.yaml"},
    {"id": "E6",  "skenario": "Banyak Kendaraan (Padat)",    "video": "video_padat.mp4",      "conf": 0.35, "line_ratio": 0.50, "tracker": "bytetrack.yaml"},
    {"id": "E7",  "skenario": "Objek Saling Berdekatan",     "video": "video_berdekatan.mp4", "conf": 0.35, "line_ratio": 0.40, "tracker": "bytetrack.yaml"},
    {"id": "E8",  "skenario": "Occlusion Ringan",            "video": "video_occlusion.mp4",  "conf": 0.35, "line_ratio": 0.40, "tracker": "bytetrack.yaml"},
    {"id": "E9",  "skenario": "Pergerakan Cepat",            "video": "video_cepat.mp4",      "conf": 0.35, "line_ratio": 0.40, "tracker": "bytetrack.yaml"},
    {"id": "E10", "skenario": "Pencahayaan Berbeda (Malam)", "video": "video_cahaya.mp4",     "conf": 0.30, "line_ratio": 0.50, "tracker": "bytetrack.yaml"},
]

VEHICLE_CLASSES = {1: "motorcycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MODEL_NAME = "yolo11s.pt" 

base_model = YOLO(MODEL_NAME)
if os.path.exists(MODEL_NAME):
    shutil.copy(MODEL_NAME, "model/detection_model.pt")

with open("tracking/tracker_config.yaml", "w") as f:
    f.write("tracker_type: bytetrack\ntrack_high_thresh: 0.5\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.6\ntrack_buffer: 30\nmatch_thresh: 0.8\n")

def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def run_single_experiment(exp):
    video_path = os.path.join(VIDEO_DIR, exp["video"])
    if not os.path.exists(video_path):
        print(f"⚠️ Dilewati [{exp['id']}]: File '{exp['video']}' tidak ditemukan.")
        return None

    print(f"▶️ Memproses {exp['id']} - {exp['skenario']} | conf={exp['conf']} | line={exp['line_ratio']}...")

    exp_model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    line_y = int(height * exp["line_ratio"])

    out_path = os.path.join(OUTPUT_VIDEO_DIR, f"{exp['id']}_{exp['video']}")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))

    track_history = defaultdict(list)
    counted_ids = set()
    unique_ids_seen = set()
    count_a_b = 0
    count_b_a = 0
    class_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    
    tracking_logs = []
    trajectory_logs = []
    frame_count = 0
    start_time = time.time()

    is_night_video = (exp["id"] == "E10")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        process_frame = enhance_low_light(frame) if is_night_video else frame

        results = exp_model.track(
            process_frame,
            persist=True,
            conf=exp["conf"],
            imgsz=1280,
            device=DEVICE,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker=exp["tracker"],
            verbose=False
        )

        display_frame = process_frame.copy()
        cv2.line(display_frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
        cv2.putText(display_frame, f"COUNT LINE ({exp['line_ratio']})", (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confs):
                unique_ids_seen.add(tid)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_history[tid].append((cx, cy))
                label = VEHICLE_CLASSES[cid]

                if exp["id"] == "E1":
                    tracking_logs.append({
                        "frame": frame_count, "track_id": tid, "class": label,
                        "confidence": round(conf, 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
                    trajectory_logs.append({
                        "track_id": tid, "frame": frame_count, "center_x": cx, "center_y": cy
                    })

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 144, 30), 2)
                cv2.putText(display_frame, f"ID {tid} | {label}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                pts = track_history[tid][-25:]
                for i in range(1, len(pts)):
                    cv2.line(display_frame, pts[i - 1], pts[i], (0, 255, 0), 2)

                if len(track_history[tid]) >= 2 and tid not in counted_ids:
                    prev_y = track_history[tid][-2][1]
                    if prev_y < line_y <= cy:
                        count_a_b += 1
                        class_counts[label] += 1
                        counted_ids.add(tid)
                    elif prev_y > line_y >= cy:
                        count_b_a += 1
                        class_counts[label] += 1
                        counted_ids.add(tid)

        total = count_a_b + count_b_a
        cv2.rectangle(display_frame, (10, 10), (420, 48), (0, 0, 0), -1)
        cv2.putText(display_frame, f"{exp['id']} | Total: {total} (A->B:{count_a_b}, B->A:{count_b_a})",
                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        out.write(display_frame)

    cap.release()
    out.release()

    if exp["id"] == "E1":
        pd.DataFrame(tracking_logs).to_csv("results/tracking_results.csv", index=False)
        pd.DataFrame(trajectory_logs).to_csv("results/trajectory.csv", index=False)
        pd.DataFrame([{
            "total_vehicles": count_a_b + count_b_a,
            "car": class_counts["car"], "motorcycle": class_counts["motorcycle"],
            "bus": class_counts["bus"], "truck": class_counts["truck"],
            "direction_A_to_B": count_a_b, "direction_B_to_A": count_b_a
        }]).to_csv("results/counting_results.csv", index=False)

    elapsed = time.time() - start_time
    fps = round(frame_count / elapsed, 2) if elapsed > 0 else 0
    inference_ms = round((elapsed / frame_count) * 1000, 2) if frame_count > 0 else 0

    return {
        "Experiment_ID": exp["id"],
        "Skenario": exp["skenario"],
        "Video_File": exp["video"],
        "Conf_Threshold": exp["conf"],
        "Line_Ratio": exp["line_ratio"],
        "Tracker": exp["tracker"],
        "Total_Counted_AI": count_a_b + count_b_a,
        "Arah_A_to_B": count_a_b,
        "Arah_B_to_A": count_b_a,
        "Total_Unique_IDs": len(unique_ids_seen),
        "Avg_FPS": fps,
        "Inference_Time_ms": inference_ms,
        "Ground_Truth_Manual": 0,
        "ID_Switch_Manual": 0
    }

all_results = []
for exp in experiments:
    res = run_single_experiment(exp)
    if res is not None:
        all_results.append(res)

if all_results:
    df_exp = pd.DataFrame(all_results)
    df_exp.to_csv("results/experiment_results.csv", index=False)
    print("\n✅ Selesai! Semua hasil telah diperbarui di folder 'results/'.")