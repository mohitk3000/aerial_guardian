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
OUTPUT     = "../output/output_kf_tracked.mp4"
MODEL_PATH = "../models/yolov8s.pt"
os.makedirs("../output", exist_ok=True)

# ── YOLO + SAHI ────────────────────────────────────────────
model = YOLO(MODEL_PATH)
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

# ── SAHI detection ─────────────────────────────────────────
def run_sahi_detection(frame_enhanced):
    sahi_result = get_sliced_prediction(
        frame_enhanced, sahi_model,
        slice_height=256, slice_width=256,
        overlap_height_ratio=0.4, overlap_width_ratio=0.4,
        verbose=0
    )
    all_boxes, all_scores = [], []
    for obj in sahi_result.object_prediction_list:
        if obj.category.name != "person":
            continue
        x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
        x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
        if not is_person_shape(x1, y1, x2, y2):
            continue
        all_boxes.append([x1, y1, x2, y2])
        all_scores.append(float(obj.score.value))
    if len(all_boxes) > 0:
        keep = nms(
            torch.tensor(all_boxes,  dtype=torch.float32),
            torch.tensor(all_scores, dtype=torch.float32),
            iou_threshold=0.4
        )
        all_boxes  = [all_boxes[i]  for i in keep]
        all_scores = [all_scores[i] for i in keep]
    return all_boxes, all_scores

# ── Kalman Filter Tracker ─────────────────────────────────
class KalmanTrack:
    """
    Single track with Kalman Filter state estimation.
    State: [cx, cy, vx, vy]
    """

    def __init__(self, track_id, cx, cy):
        self.track_id  = track_id
        self.lost      = 0          # frames since last detection
        self.hits      = 1          # total detections matched

        # ── State vector: [cx, cy, vx, vy] ────────────────
        # Start with zero velocity — we don't know direction yet
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)

        # ── State transition matrix F ──────────────────────
        # Describes how state evolves each frame
        # new_cx = cx + vx*dt  (dt=1 frame)
        # new_cy = cy + vy*dt
        # new_vx = vx          (constant velocity)
        # new_vy = vy
        self.F = np.array([
            [1, 0, 1, 0],   # cx = cx + vx
            [0, 1, 0, 1],   # cy = cy + vy
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1]    # vy = vy
        ], dtype=np.float32)

        # ── Measurement matrix H ───────────────────────────
        # We only OBSERVE position (cx, cy), not velocity
        # Maps state [cx,cy,vx,vy] → measurement [cx,cy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # ── Process noise Q ────────────────────────────────
        # How much we trust our motion model
        # Higher = more uncertain about predicted position
        # Drone moves smoothly → low process noise
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.Q[2, 2] = 0.5   # slightly more uncertainty on velocity
        self.Q[3, 3] = 0.5

        # ── Measurement noise R ────────────────────────────
        # How much we trust detections
        # Higher = less trust in raw detections
        # SAHI detections are fairly reliable → low R
        self.R = np.eye(2, dtype=np.float32) * 5.0

        # ── Error covariance P ─────────────────────────────
        # Uncertainty in our state estimate
        # Starts high (we just initialized), decreases as we get detections
        self.P = np.eye(4, dtype=np.float32) * 100.0

    def predict(self):
        """
        Step 1 of KF: Predict next state using motion model
        Called every frame BEFORE matching detections
        Returns predicted (cx, cy)
        """
        # Predicted state
        self.state = self.F @ self.state

        # Predicted covariance (uncertainty grows without measurement)
        self.P = self.F @ self.P @ self.F.T + self.Q

        return int(self.state[0]), int(self.state[1])

    def update(self, cx, cy):
        """
        Step 2 of KF: Update state with actual detection
        Called when a detection is matched to this track
        Blends predicted position with measured position
        """
        measurement = np.array([cx, cy], dtype=np.float32)

        # Innovation: difference between measurement and prediction
        y = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: how much to trust measurement vs prediction
        # High gain = trust measurement more
        # Low gain  = trust prediction more
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state: prediction + gain * innovation
        self.state = self.state + K @ y

        # Updated covariance: uncertainty decreases after measurement
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.lost  = 0
        self.hits += 1

    def get_position(self):
        """Returns current estimated (cx, cy)"""
        return int(self.state[0]), int(self.state[1])


class KalmanCentroidTracker:
    """
    Multi-object tracker using Kalman Filters.
    Each tracked person has their own KalmanTrack instance.

    Matching strategy:
    1. Predict all tracks forward one frame
    2. Match detections to predicted positions by distance
    3. Update matched tracks with actual detection
    4. Keep unmatched tracks alive for TRACK_BUFFER frames
    5. Create new tracks for unmatched detections
    """

    def __init__(self, max_dist=80, track_buffer=15):
        self.tracks      = {}   # {track_id: KalmanTrack}
        self.next_id     = 0
        self.max_dist    = max_dist
        self.track_buffer= track_buffer

    def update(self, boxes):
        """
        Main update function called every frame.
        Returns list of (track_id, cx, cy) for active tracks.
        """

        # ── Step 1: Predict all tracks forward ────────────
        predicted = {}
        for tid, track in self.tracks.items():
            px, py = track.predict()
            predicted[tid] = (px, py)

        # ── Step 2: Get detection centroids ───────────────
        detections = [
            ((b[0]+b[2])//2, (b[1]+b[3])//2)
            for b in boxes
        ]

        # ── Step 3: Match detections to predicted positions
        # Uses greedy nearest-neighbour matching
        matched_det  = set()
        matched_track= set()

        # Build distance matrix
        matches = []
        for di, (dcx, dcy) in enumerate(detections):
            for tid, (px, py) in predicted.items():
                dist = np.sqrt((dcx-px)**2 + (dcy-py)**2)
                if dist < self.max_dist:
                    matches.append((dist, di, tid))

        # Sort by distance — greedily assign closest pairs first
        matches.sort(key=lambda x: x[0])
        for dist, di, tid in matches:
            if di in matched_det or tid in matched_track:
                continue
            # ── Update matched track with real detection ───
            dcx, dcy = detections[di]
            self.tracks[tid].update(dcx, dcy)
            matched_det.add(di)
            matched_track.add(tid)

        # ── Step 4: Create new tracks for unmatched detections
        for di, (dcx, dcy) in enumerate(detections):
            if di not in matched_det:
                self.tracks[self.next_id] = KalmanTrack(self.next_id, dcx, dcy)
                self.next_id += 1

        # ── Step 5: Handle unmatched tracks ───────────────
        to_delete = []
        for tid in self.tracks:
            if tid not in matched_track:
                self.tracks[tid].lost += 1
                if self.tracks[tid].lost > self.track_buffer:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        # ── Return active tracks (lost=0 only) ────────────
        active = []
        for tid, track in self.tracks.items():
            if track.lost == 0:
                cx, cy = track.get_position()
                active.append((tid, cx, cy))

        return active


# ── Trajectory storage ─────────────────────────────────────
track_history = defaultdict(list)
TAIL_LENGTH   = 60

def get_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(50, 255, 3).tolist())

# ── Frames ─────────────────────────────────────────────────
frames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"Total frames : {len(frames)}")

first = cv2.imread(os.path.join(IMG_DIR, frames[0]))
h, w  = first.shape[:2]
out   = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

# ── Initialize KF tracker ──────────────────────────────────
tracker      = KalmanCentroidTracker(max_dist=80, track_buffer=15)
DETECT_EVERY = 2
last_boxes   = []
fps_list     = []

for i, fname in enumerate(frames):
    frame          = cv2.imread(os.path.join(IMG_DIR, fname))
    frame_enhanced = apply_clahe(frame)
    annotated      = frame.copy()
    t0             = time.time()

    # ── Frame skip ─────────────────────────────────────────
    if i % DETECT_EVERY == 0 or i == 0:
        boxes, scores = run_sahi_detection(frame_enhanced)
        last_boxes    = boxes
    else:
        boxes = last_boxes

    # ── KF Tracker update ──────────────────────────────────
    active_tracks = tracker.update(boxes)

    fps = 1 / (time.time() - t0)
    fps_list.append(fps)

    # ── Draw boxes + tails ─────────────────────────────────
    for box, (track_id, cx, cy) in zip(boxes, active_tracks):
        x1, y1, x2, y2 = box
        color           = get_color(track_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"P{track_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Use KF predicted position for tail (smoother)
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > TAIL_LENGTH:
            track_history[track_id].pop(0)

        points = track_history[track_id]
        for j in range(1, len(points)):
            thickness = max(1, int(4 * j / len(points)))
            cv2.line(annotated, points[j-1], points[j], color, thickness)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

        # ── Draw predicted position as small cross ─────────
        # Shows where KF thinks person will be next frame
        px, py = tracker.tracks[track_id].predict()
        # restore state after peeking (re-update immediately)
        tracker.tracks[track_id].update(cx, cy)
        cv2.drawMarker(annotated, (px, py), color,
                       cv2.MARKER_CROSS, 10, 1)

    # ── Overlays ───────────────────────────────────────────
    cv2.putText(annotated, f"FPS: {fps:.1f}",                        (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 0),        2)
    cv2.putText(annotated, f"Persons: {len(active_tracks)}",         (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255),      2)
    cv2.putText(annotated, "CLAHE+SAHI+AspectFilter+KalmanTracker",  (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),      2)

    out.write(annotated)

    if i % 50 == 0:
        print(f"Frame {i}/{len(frames)} | FPS: {fps:.1f} | Tracked: {len(active_tracks)}")

out.release()
avg_fps = sum(fps_list) / len(fps_list)
print(f"\nDone! Saved to {OUTPUT}")
print(f"Average FPS : {avg_fps:.2f}")
print(f"Hardware    : {torch.cuda.get_device_name(0)}")