import cv2
import time
import os
from collections import defaultdict
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Paths ──────────────────────────────────────────────────
SEQ_PATH   = "../dataset/VisDrone-MOT-val/sequences/uav0000086_00000_v"
IMG_DIR    = SEQ_PATH
OUTPUT     = "../output/output_tracked.mp4"
MODEL_PATH = "../models/yolov8s.pt"
os.makedirs("../output", exist_ok=True)

# ── YOLO model ─────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ── SAHI model (FP16 enabled) ──────────────────────────────
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=0.20,  
    device="cuda", 
    image_size=256 
)

# ── CLAHE ──────────────────────────────────────────────────
# Applied in LAB space — only L channel enhanced
# Fixes drone haze without distorting colors
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ── Aspect ratio filter ────────────────────────────────────
# Persons: h/w > 0.75 (tall or square)
# Bicycles/cars: h/w < 0.75 (wide) → rejected
ASPECT_RATIO = 0.75  # h/w > 0.75 = person, else reject (bikes/cars)
def is_person_shape(x1, y1, x2, y2):
    bw = x2 - x1
    bh = y2 - y1
    if bw == 0 or bh == 0:
        return False
    return (bh / bw) > ASPECT_RATIO

# ── SAHI tiled detection ───────────────────────────────────
# Slices frame into 256x256 tiles with 40% overlap
# Maps detections back to full frame coordinates
# Runs NMS to remove duplicates from overlapping tiles
def run_sahi_detection(frame_enhanced):
    sahi_result = get_sliced_prediction(
        frame_enhanced,
        sahi_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
        verbose=0
    )

    all_boxes  = []
    all_scores = []

    for obj in sahi_result.object_prediction_list:
        if obj.category.name != "person":
            continue

        x1 = int(obj.bbox.minx)
        y1 = int(obj.bbox.miny)
        x2 = int(obj.bbox.maxx)
        y2 = int(obj.bbox.maxy)

        # Aspect ratio filter — removes bicycle
        if not is_person_shape(x1, y1, x2, y2):
            continue

        all_boxes.append([x1, y1, x2, y2])
        all_scores.append(float(obj.score.value))

    # Global NMS across tiles — removes duplicate boxes
    if len(all_boxes) > 0:
        boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        keep     = nms(boxes_t, scores_t, iou_threshold=0.4)
        all_boxes  = [all_boxes[i]  for i in keep]
        all_scores = [all_scores[i] for i in keep]

    return all_boxes, all_scores

# ── Trajectory storage ─────────────────────────────────────
track_history = defaultdict(list)
TAIL_LENGTH   = 60

def get_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(50, 255, 3).tolist())

# ── Centroid tracker with buffer ───────────────────────────
# Keeps lost IDs alive for TRACK_BUFFER frames
# Handles brief occlusions from drone motion
next_id       = 0
active_tracks = {}   # {id: (cx, cy, frames_lost)}
TRACK_BUFFER  = 15

def update_tracker(boxes, max_dist=60):
    global next_id, active_tracks

    centroids  = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes]
    new_tracks = {}
    used_ids   = set()

    for cx, cy in centroids:
        best_id, best_dist = None, max_dist
        for tid, (px, py, lost) in active_tracks.items():
            d = np.sqrt((cx-px)**2 + (cy-py)**2)
            if d < best_dist and tid not in used_ids:
                best_dist, best_id = d, tid
        if best_id is not None:
            new_tracks[best_id] = (cx, cy, 0)
            used_ids.add(best_id)
        else:
            new_tracks[next_id] = (cx, cy, 0)
            next_id += 1

    # Keep recently lost tracks alive (track buffer)
    for tid, (px, py, lost) in active_tracks.items():
        if tid not in new_tracks and lost < TRACK_BUFFER:
            new_tracks[tid] = (px, py, lost + 1)

    active_tracks = new_tracks
    return [tid for tid, (*_, lost) in active_tracks.items() if lost == 0]

# ── Frames ─────────────────────────────────────────────────
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"Total frames : {len(frames)}")

first = cv2.imread(os.path.join(IMG_DIR, frames[0]))
h, w  = first.shape[:2]
out   = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

# ── Frame skip settings ────────────────────────────────────
DETECT_EVERY = 2   # run SAHI every 2nd frame
last_boxes   = []
last_scores  = []
fps_list     = []

# Measure total wall time
total_start = time.time()

# ── Main loop ──────────────────────────────────────────────
for i, fname in enumerate(frames):
    frame          = cv2.imread(os.path.join(IMG_DIR, fname))
    frame_enhanced = apply_clahe(frame)   # CLAHE on every frame
    annotated      = frame.copy()
    t0             = time.time()

    # ── Frame skip: SAHI only on even frames ───────────────
    if i % DETECT_EVERY == 0 or i == 0:
        boxes, scores = run_sahi_detection(frame_enhanced)
        last_boxes    = boxes
        last_scores   = scores
    else:
        boxes  = last_boxes    # reuse cached detections
        scores = last_scores

    # ── Update tracker with current detections ─────────────
    track_ids = update_tracker(boxes)

    fps = 1 / (time.time() - t0)
    fps_list.append(fps)

    # ── Draw boxes + tails ─────────────────────────────────
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box
        color           = get_color(track_id)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # ID label
        cv2.putText(annotated, f"P{track_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trajectory tail (fading, thickness-graduated)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > TAIL_LENGTH:
            track_history[track_id].pop(0)

        points = track_history[track_id]
        for j in range(1, len(points)):
            thickness = max(1, int(4 * j / len(points)))
            cv2.line(annotated, points[j-1], points[j], color, thickness)
        cv2.circle(annotated, (cx, cy), 3, color, -1)

    # ── Overlays ───────────────────────────────────────────
    cv2.putText(annotated, f"FPS: {fps:.1f}",                   (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),   2)
    cv2.putText(annotated, f"Persons: {len(boxes)}",            (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255), 2)
    cv2.putText(annotated, "CLAHE+SAHI+AspectFilter+CentroidTracker", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    out.write(annotated)

    if i % 50 == 0:
        print(f"Frame {i}/{len(frames)} | FPS: {fps:.1f} | Persons: {len(boxes)}")

out.release()
total_end  = time.time()
total_time = total_end - total_start
avg_fps   = len(frames) / total_time
print(f"Avg. FPS    : {avg_fps:.2f}")
print(f"\nDone! Saved to {OUTPUT}")

# ============================================================
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"Hardware     : {torch.cuda.get_device_name(0)}")