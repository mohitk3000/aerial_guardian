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
SEQ_ROOT   = "../dataset/VisDrone-MOT-val/sequences"
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
    return (bh / bw) > 0.75

# ── SAHI tiled detection ───────────────────────────────────
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

# ── Color helper ───────────────────────────────────────────
def get_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(50, 255, 3).tolist())

# ══════════════════════════════════════════════════════════
# TRACKER — reset per sequence
# Each sequence is independent — IDs must restart fresh
# ══════════════════════════════════════════════════════════
def make_tracker():
    """Returns a fresh tracker state for each sequence"""
    return {
        "next_id":       0,
        "active_tracks": {}   # {id: (cx, cy, frames_lost)}
    }

def update_tracker(tracker, boxes, max_dist=60):
    next_id       = tracker["next_id"]
    active_tracks = tracker["active_tracks"]
    TRACK_BUFFER  = 15

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

    tracker["next_id"]       = next_id
    tracker["active_tracks"] = new_tracks
    return [tid for tid, (*_, lost) in new_tracks.items() if lost == 0]

# ══════════════════════════════════════════════════════════
# MAIN — loop over ALL sequences
# ══════════════════════════════════════════════════════════
sequences    = sorted(os.listdir(SEQ_ROOT))
DETECT_EVERY = 2
TAIL_LENGTH  = 60

total_start      = time.time()
all_fps          = []
seq_summary      = []   # store per-sequence stats for final report

print(f"Found {len(sequences)} sequences")
print("=" * 60)

for seq_idx, seq in enumerate(sequences):
    SEQ_PATH = os.path.join(SEQ_ROOT, seq)

    # Skip if not a directory
    if not os.path.isdir(SEQ_PATH):
        continue

    frames = sorted([f for f in os.listdir(SEQ_PATH) if f.endswith(".jpg")])
    if len(frames) == 0:
        print(f"Skipping {seq} — no frames found")
        continue

    OUTPUT = f"../output/{seq}_tracked.mp4"
    print(f"\n[{seq_idx+1}/{len(sequences)}] Processing: {seq}")
    print(f"  Frames: {len(frames)}")

    # ── Read first frame to get dimensions ─────────────────
    first = cv2.imread(os.path.join(SEQ_PATH, frames[0]))
    h, w  = first.shape[:2]

    # ── Fresh video writer per sequence ────────────────────
    out = cv2.VideoWriter(
        OUTPUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h)
    )

    # ── Reset ALL state per sequence ───────────────────────
    tracker       = make_tracker()          # fresh IDs
    track_history = defaultdict(list)       # fresh tails
    last_boxes    = []
    last_scores   = []
    fps_list      = []
    seq_start     = time.time()

    for i, fname in enumerate(frames):
        frame          = cv2.imread(os.path.join(SEQ_PATH, fname))
        frame_enhanced = apply_clahe(frame)
        annotated      = frame.copy()
        t0             = time.time()

        # ── Frame skip ─────────────────────────────────────
        if i % DETECT_EVERY == 0 or i == 0:
            boxes, scores = run_sahi_detection(frame_enhanced)
            last_boxes    = boxes
            last_scores   = scores
        else:
            boxes  = last_boxes
            scores = last_scores

        # ── Track ──────────────────────────────────────────
        track_ids = update_tracker(tracker, boxes)

        fps = 1 / (time.time() - t0)
        fps_list.append(fps)
        all_fps.append(fps)

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
            for j in range(1, len(points)):
                thickness = max(1, int(4 * j / len(points)))
                cv2.line(annotated, points[j-1], points[j], color, thickness)
            cv2.circle(annotated, (cx, cy), 3, color, -1)

        # ── Overlays ───────────────────────────────────────
        cv2.putText(annotated, f"Seq: {seq}",                       (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f}",                   (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),     2)
        cv2.putText(annotated, f"Persons: {len(boxes)}",            (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255),   2)
        cv2.putText(annotated, "CLAHE+SAHI+AspectFilter+CentroidTracker", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),   2)

        out.write(annotated)

        if i % 50 == 0:
            print(f"  Frame {i}/{len(frames)} | FPS: {fps:.1f} | Persons: {len(boxes)}")

    # ── End of sequence ────────────────────────────────────
    out.release()
    seq_time    = time.time() - seq_start
    seq_avg_fps = len(frames) / seq_time
    seq_summary.append({
        "seq":      seq,
        "frames":   len(frames),
        "avg_fps":  seq_avg_fps,
        "output":   OUTPUT
    })
    print(f"  Done! → {OUTPUT}")
    print(f"  Avg FPS: {seq_avg_fps:.2f} | Time: {seq_time:.1f}s")

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
total_time     = time.time() - total_start
overall_fps    = sum(s["frames"] for s in seq_summary) / total_time

print("\n" + "=" * 60)
print("ALL SEQUENCES COMPLETE")
print("=" * 60)
for s in seq_summary:
    print(f"  {s['seq']:40s} | {s['frames']:4d} frames | {s['avg_fps']:.2f} FPS")
print("-" * 60)
print(f"  Total sequences : {len(seq_summary)}")
print(f"  Total frames    : {sum(s['frames'] for s in seq_summary)}")
print(f"  Overall avg FPS : {overall_fps:.2f}")
print(f"  Total time      : {total_time/60:.1f} mins")
print(f"  Hardware        : {torch.cuda.get_device_name(0)}")
print("=" * 60)