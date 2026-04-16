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
OUTPUT     = "../output/output_tracked_batch.mp4"
MODEL_PATH = "../models/yolov8s.pt"
os.makedirs("../output", exist_ok=True)

# ── YOLO model ─────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ── SAHI model ─────────────────────────────────────────────
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=0.20,
    device="cuda",  
    image_size=256 
)

# ══════════════════════════════════════════════════════════
# BATCH DETECTION
# Processes BATCH_SIZE frames in one GPU call
# Much faster than one frame at a time for detection
# tracking still runs sequentially (ByteTrack needs order)
# process 4 frames per GPU call
# increase to 8 if you have VRAM headroom
# decrease to 2 if you get OOM errors
# ══════════════════════════════════════════════════════════
BATCH_SIZE = 4 

ASPECT_RATIO = 0.75  # h/w > 0.75 = person, else reject (bikes/cars)

# ── CLAHE ──────────────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ── Aspect ratio filter ────────────────────────────────────
def is_person_shape(x1, y1, x2, y2):
    bw = x2 - x1
    bh = y2 - y1
    if bw == 0 or bh == 0:
        return False
    return (bh / bw) > ASPECT_RATIO

# ── SAHI tiled detection (single frame) ───────────────────
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
        if not is_person_shape(x1, y1, x2, y2):
            continue
        all_boxes.append([x1, y1, x2, y2])
        all_scores.append(float(obj.score.value))

    if len(all_boxes) > 0:
        boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        keep     = nms(boxes_t, scores_t, iou_threshold=0.4)
        all_boxes  = [all_boxes[i]  for i in keep]
        all_scores = [all_scores[i] for i in keep]

    return all_boxes, all_scores

def run_batch_sahi(frame_list):
    """
    Run SAHI on a batch of frames
    Returns list of (boxes, scores) per frame
    """
    batch_results = []
    for frame in frame_list:
        boxes, scores = run_sahi_detection(frame)
        batch_results.append((boxes, scores))
    return batch_results

def run_batch_yolo(frame_list):
    """
    Run YOLO on multiple frames in ONE GPU call
    This is true batching — all frames processed simultaneously
    Used for fast pre-screening before SAHI
    """
    results = model(
        frame_list,        # ← list of frames = batch
        classes=[0],
        imgsz=640,         # smaller size for batch speed
        conf=0.20,
        half=True,
        verbose=False
    )
    return results

# ── Trajectory storage ─────────────────────────────────────
track_history = defaultdict(list)
TAIL_LENGTH   = 60

def get_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(50, 255, 3).tolist())

# ── Centroid tracker with buffer ───────────────────────────
next_id       = 0
active_tracks = {}
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

    for tid, (px, py, lost) in active_tracks.items():
        if tid not in new_tracks and lost < TRACK_BUFFER:
            new_tracks[tid] = (px, py, lost + 1)

    active_tracks = new_tracks
    return [tid for tid, (*_, lost) in active_tracks.items() if lost == 0]

# ── Frames ─────────────────────────────────────────────────
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"Total frames : {len(frames)}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Total batches: {len(frames) // BATCH_SIZE}")

first = cv2.imread(os.path.join(IMG_DIR, frames[0]))
h, w  = first.shape[:2]
out   = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

fps_list     = []
DETECT_EVERY = 2
last_boxes   = []
last_scores  = []

# Measure total wall time
total_start = time.time()

# ── Main loop — process in batches ────────────────────────
# Outer loop: chunks of BATCH_SIZE frames
# Inner loop: track each frame sequentially

for batch_start in range(0, len(frames), BATCH_SIZE):

    # ── Load batch of frames ───────────────────────────────
    batch_fnames   = frames[batch_start : batch_start + BATCH_SIZE]
    batch_frames   = [cv2.imread(os.path.join(IMG_DIR, f)) for f in batch_fnames]
    batch_enhanced = [apply_clahe(f) for f in batch_frames]

    # ── YOLO batch pre-screen ──────────────────────────────
    # Run fast YOLO on all frames at once to find which
    # frames actually have persons → only run SAHI on those
    t_batch = time.time()
    yolo_batch_results = run_batch_yolo(batch_enhanced)
    batch_prescreen_time = time.time() - t_batch

    # ── Process each frame in batch sequentially ───────────
    for j, (frame, frame_enhanced, fname) in enumerate(
        zip(batch_frames, batch_enhanced, batch_fnames)
    ):
        global_i = batch_start + j
        t0       = time.time()
        annotated = frame.copy()

        # ── Check if YOLO pre-screen found any persons ─────
        yolo_result    = yolo_batch_results[j]
        has_detections = (
            yolo_result.boxes is not None and
            len(yolo_result.boxes) > 0
        )

        # ── Smart detection strategy ───────────────────────
        # If YOLO found persons AND it's a detection frame:
        #   → run full SAHI for maximum recall
        # If YOLO found nothing:
        #   → skip SAHI entirely (saves time on empty frames)
        # If it's a skipped frame:
        #   → reuse last detections

        if global_i % DETECT_EVERY == 0 or global_i == 0:
            if has_detections:
                # Full SAHI — catches tiny/seated persons
                boxes, scores = run_sahi_detection(frame_enhanced)
            else:
                # YOLO found nothing → skip expensive SAHI
                boxes, scores = [], []
            last_boxes  = boxes
            last_scores = scores
        else:
            # Frame skip — reuse cached detections
            boxes  = last_boxes
            scores = last_scores

        # ── Update tracker ─────────────────────────────────
        track_ids = update_tracker(boxes)

        fps = 1 / (time.time() - t0)
        fps_list.append(fps)

        # ── Draw boxes + tails ─────────────────────────────
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            color           = get_color(track_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"P{track_id}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cx, cy = (x1+x2)//2, (y1+y2)//2
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > TAIL_LENGTH:
                track_history[track_id].pop(0)

            points = track_history[track_id]
            for k in range(1, len(points)):
                thickness = max(1, int(4 * k / len(points)))
                cv2.line(annotated, points[k-1], points[k], color, thickness)
            cv2.circle(annotated, (cx, cy), 3, color, -1)

        # ── Overlays ───────────────────────────────────────
        cv2.putText(annotated, f"FPS: {fps:.1f}",                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),   2)
        cv2.putText(annotated, f"Persons: {len(boxes)}",             (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255), 2)
        cv2.putText(annotated, f"Batch={BATCH_SIZE} CLAHE+SAHI+AspectFilter+CentroidTracker", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        out.write(annotated)

        if global_i % 50 == 0:
            print(f"Frame {global_i}/{len(frames)} | FPS: {fps:.1f} | Persons: {len(boxes)} | SAHI: {has_detections}")

out.release()
total_end  = time.time()
total_time = total_end - total_start
avg_fps   = len(frames) / total_time
print(f"\nDone! Saved to {OUTPUT}")
print(f"Average FPS  : {avg_fps:.2f}")
print(f"Batch size   : {BATCH_SIZE}")

# ============================================================
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"Hardware     : {torch.cuda.get_device_name(0)}")