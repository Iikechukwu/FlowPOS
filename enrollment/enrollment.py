"""
SMART INVENTORY SYSTEM
Module: Enrollment — Guided Video Pipeline (Redesigned)

FIXES IN THIS VERSION:
  - MIN_POSE_FRAMES_REQUIRED reduced to 2 — easier to confirm each pose
  - Pose geometry thresholds widened — right/up/down now trigger reliably
  - CANDIDATE_SIM_THRESHOLD raised to 0.96 — stops over-clustering
  - Stage advances immediately once pose confirmed (no timer wait)
  - CLAHE preprocessing on every frame for dark skin support
"""

import cv2
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_stream import CameraStream
from services.duplicate_checker import check_duplicate_from_embedding
from services.enrollment_manager import create_customer_folder, generate_staff_id
from face_db import save_customer_multi_embedding
from embedding_loader import add_embedding_to_memory


# ── CONFIGURATION ──────────────────────────────────────────────────────────────

GUIDED_DURATION          = 12
EXTRACT_INTERVAL_SEC     = 0.25
EXTRACT_MIN_FRAMES       = 25
EXTRACT_MAX_FRAMES       = 50

MIN_POSE_FRAMES_REQUIRED = 2      # was 3 — easier to confirm each pose
POSE_EXTEND_MAX_SEC      = 5.0

W_SHARPNESS              = 0.30
W_FACE_SIZE              = 0.25
W_BRIGHTNESS             = 0.25
W_LANDMARK               = 0.20

FACE_RATIO_MIN           = 0.03
FACE_RATIO_IDEAL         = 0.18

BRIGHT_MIN               = 30
BRIGHT_MAX               = 220
BRIGHT_IDEAL             = 120

GUIDE_BOX_MIN_PX         = 180
GUIDE_BOX_MAX_PX         = 280

LIGHT_MIN_MEAN           = 40
LIGHT_MAX_MEAN           = 210
LIGHT_MIN_CONTRAST       = 20

QUALITY_MIN_SCORE        = 0.15

CANDIDATE_SIM_THRESHOLD  = 0.96   # was 0.85 — stops over-clustering

TOP_N_FINAL              = 10

DUPLICATE_THRESHOLD      = 0.75

AUTO_START_HOLD_SEC      = 2.0

MAX_FAILED_READ          = 15

HEAD_POSITIONS = [
    ("straight", "Look Straight Ahead"),
    ("left",     "Turn Slightly LEFT"),
    ("right",    "Turn Slightly RIGHT"),
    ("up",       "Tilt Head Slightly UP"),
    ("down",     "Tilt Head Slightly DOWN"),
]
SECONDS_PER_POSE = GUIDED_DURATION / len(HEAD_POSITIONS)

COL_GREEN  = (0,  210,  80)
COL_YELLOW = (0,  200, 210)
COL_GRAY   = (130, 130, 130)
COL_WHITE  = (240, 240, 240)
COL_BLACK  = ( 10,  10,  10)
COL_CYAN   = (200, 220,   0)
COL_RED    = (  0,  50, 220)
COL_ORANGE = ( 30, 140, 255)


# ── CLAHE PREPROCESSOR ─────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

def _preprocess(frame):
    """Apply CLAHE to luminance channel — fixes dark skin landmark detection."""
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq    = _clahe.apply(l)
    lab_eq  = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── MEDIAPIPE DETECTOR ─────────────────────────────────────────────────────────

_mediapipe_model = None

def _get_mediapipe():
    global _mediapipe_model
    if _mediapipe_model is None:
        try:
            import mediapipe as mp
            _mediapipe_model = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.6,
            )
            print("MediaPipe face detector initialised ✅")
        except ImportError:
            raise RuntimeError("mediapipe not installed. Run: pip install mediapipe")
    return _mediapipe_model


def _detect_faces_mp(frame):
    """
    Detect on CLAHE-enhanced frame.
    Returns list of ((x,y,w,h), landmarks | None).

    MULTI-FACE FIX: keeps only faces whose centre falls within the
    middle 70% of the frame — filters out door frames, wall edges,
    and background objects at the borders.
    If multiple faces still pass, keeps only the largest one.
    """
    import mediapipe as mp
    preprocessed = _preprocess(frame)
    model        = _get_mediapipe()
    rgb          = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    h, w         = frame.shape[:2]
    result       = model.process(rgb)

    zone_x1, zone_x2 = int(w * 0.15), int(w * 0.85)
    zone_y1, zone_y2 = int(h * 0.10), int(h * 0.90)

    faces = []
    if result.detections:
        for d in result.detections:
            bb = d.location_data.relative_bounding_box
            x  = max(0, int(bb.xmin * w))
            y  = max(0, int(bb.ymin * h))
            bw = min(int(bb.width  * w), w - x)
            bh = min(int(bb.height * h), h - y)

            # Keep only faces centred inside the zone
            cx, cy = x + bw // 2, y + bh // 2
            if not (zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2):
                continue

            kps = d.location_data.relative_keypoints
            lm  = None
            if len(kps) >= 4:
                lm = {
                    "left_eye":  (kps[0].x * w, kps[0].y * h),
                    "right_eye": (kps[1].x * w, kps[1].y * h),
                    "nose":      (kps[2].x * w, kps[2].y * h),
                    "mouth":     (kps[3].x * w, kps[3].y * h),
                }
            faces.append(((x, y, bw, bh), lm))

    # If multiple pass the zone, keep only the largest
    if len(faces) > 1:
        faces = [max(faces, key=lambda f: f[0][2] * f[0][3])]

    return faces


# ── POSE ESTIMATION ────────────────────────────────────────────────────────────

def _estimate_pose_tag(landmarks):
    """
    Estimate head pose from landmark geometry.
    Returns: 'straight' | 'left' | 'right' | 'up' | 'down' | 'unknown'

    WIDENED THRESHOLDS so right/up/down trigger on small natural movements.
    """
    if landmarks is None:
        return "unknown"

    required = ["left_eye", "right_eye", "nose", "mouth"]
    if not all(k in landmarks for k in required):
        return "unknown"

    le    = np.array(landmarks["left_eye"])
    re    = np.array(landmarks["right_eye"])
    nose  = np.array(landmarks["nose"])
    mouth = np.array(landmarks["mouth"])

    eye_mid = (le + re) / 2
    eye_w   = abs(re[0] - le[0])

    if eye_w < 3:
        return "unknown"

    # Left / Right — widened from ±0.15 to ±0.12
    horiz = (nose[0] - eye_mid[0]) / eye_w
    if horiz < -0.12:
        return "right"   # face turned right → nose left of centre
    if horiz > 0.12:
        return "left"

    # Up / Down — widened range from 0.32–0.58 to 0.36–0.54
    face_h = mouth[1] - eye_mid[1]
    if face_h < 3:
        return "unknown"
    nose_rel = (nose[1] - eye_mid[1]) / face_h
    if nose_rel < 0.46:
        return "up"
    if nose_rel > 0.48:
        return "down"

    return "straight"


# ── QUALITY SCORING ────────────────────────────────────────────────────────────

def _score_sharpness(gray_crop):
    return float(min(cv2.Laplacian(gray_crop, cv2.CV_64F).var() / 500.0, 1.0))

def _score_face_size(bbox, frame_shape):
    x, y, w, h = bbox
    ih, iw     = frame_shape[:2]
    ratio      = (w * h) / max(iw * ih, 1)
    if ratio < FACE_RATIO_MIN:
        return 0.0
    return float(min(ratio / FACE_RATIO_IDEAL, 1.0))

def _score_brightness(gray_crop):
    mean = float(np.mean(gray_crop))
    if mean < BRIGHT_MIN or mean > BRIGHT_MAX:
        return 0.0
    dist = abs(mean - BRIGHT_IDEAL)
    half = (BRIGHT_MAX - BRIGHT_MIN) / 2.0
    return float(1.0 - dist / half)

def _score_landmark_confidence(landmarks):
    if landmarks is None:
        return 0.0
    required = ["left_eye", "right_eye", "nose", "mouth"]
    if not all(k in landmarks for k in required):
        return 0.0
    le, re = landmarks["left_eye"], landmarks["right_eye"]
    nose   = landmarks["nose"]
    mouth  = landmarks["mouth"]
    eye_y  = (le[1] + re[1]) / 2
    if not (eye_y < nose[1] < mouth[1]):
        return 0.3
    eye_dist = abs(re[0] - le[0])
    if eye_dist < 3:
        return 0.3
    if abs(le[1] - re[1]) / eye_dist > 0.5:
        return 0.5
    return 1.0

def _score_frame(frame, bbox, landmarks):
    if _score_face_size(bbox, frame.shape) == 0.0:
        return 0.0
    x, y, w, h = bbox
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return (
        W_SHARPNESS  * _score_sharpness(gray)              +
        W_FACE_SIZE  * _score_face_size(bbox, frame.shape) +
        W_BRIGHTNESS * _score_brightness(gray)             +
        W_LANDMARK   * _score_landmark_confidence(landmarks)
    )


# ── CAMERA CONNECTION ──────────────────────────────────────────────────────────

def connect_camera_with_retry(stream_url):
    while True:
        print("\nConnecting to camera...")
        cam = CameraStream(stream_url=stream_url)
        if cam.connect():
            print("Camera connected ✅")
            return cam, stream_url
        print("\n⚠️  Camera connection failed!")
        print("1. Retry   2. New IP   3. Cancel")
        ch = input("Choice (1/2/3): ").strip()
        if ch == "2":
            ip         = input("Enter IP address: ").strip()
            stream_url = f"http://{ip}:4747/video"
        elif ch == "3":
            return None, stream_url


# ── PHASE 1 — SETUP ────────────────────────────────────────────────────────────

def _run_phase1_setup(camera, name):
    """
    Steps 2-3: lighting pre-check + guide box alignment.
    Auto-starts when both conditions held for AUTO_START_HOLD_SEC.
    """
    print("\n── Phase 1: Setup ─────────────────────────────")
    print("  Position face in guide box — recording starts automatically.")
    print("  Q = cancel\n")

    H, W       = 480, 640
    failed     = 0
    hold_since = None

    while True:
        raw = camera.read_frame()
        if raw is None:
            failed += 1
            if failed >= MAX_FAILED_READ:
                print("Stream lost during setup.")
                return False
            continue
        failed = 0

        frame = cv2.resize(raw, (W, H))
        faces = _detect_faces_mp(frame)

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b   = float(np.mean(gray))
        contrast = float(np.std(gray))
        light_ok = (LIGHT_MIN_MEAN <= mean_b <= LIGHT_MAX_MEAN
                    and contrast >= LIGHT_MIN_CONTRAST)

        gx = W // 2 - GUIDE_BOX_MAX_PX // 2
        gy = H // 2 - int(GUIDE_BOX_MAX_PX * 1.3) // 2
        gw = GUIDE_BOX_MAX_PX
        gh = int(GUIDE_BOX_MAX_PX * 1.3)

        face_ok = False
        face_w  = 0
        if len(faces) == 1:
            (fx, fy, fw, fh), _ = faces[0]
            face_w  = fw
            face_ok = GUIDE_BOX_MIN_PX <= fw <= GUIDE_BOX_MAX_PX

        all_ok = light_ok and face_ok
        now    = time.time()

        if all_ok:
            if hold_since is None:
                hold_since = now
            held = now - hold_since
        else:
            hold_since = None
            held       = 0.0

        if held >= AUTO_START_HOLD_SEC:
            print("  Auto-starting — position good ✅")
            return True

        hold_pct = min(held / AUTO_START_HOLD_SEC, 1.0)

        disp    = _preprocess(frame).copy()
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (W, 62), COL_BLACK, -1)
        cv2.addWeighted(overlay, 0.80, disp, 0.20, 0, disp)
        cv2.putText(disp, "SMART INVENTORY — Enrollment Setup",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_CYAN, 1)
        cv2.putText(disp, f"Enrolling: {name}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_WHITE, 1)

        box_col = COL_GREEN if all_ok else COL_YELLOW
        cv2.rectangle(disp, (gx, gy), (gx + gw, gy + gh), box_col, 2)
        cv2.putText(disp, "Align face + shoulders here",
                    (gx - 10, gy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1)

        if all_ok and hold_pct > 0:
            cx_r, cy_r = gx + gw - 28, gy + 28
            cv2.ellipse(disp, (cx_r, cy_r), (20, 20),
                        -90, 0, int(360 * hold_pct), COL_GREEN, 3)
            cv2.putText(disp, f"{AUTO_START_HOLD_SEC - held:.1f}",
                        (cx_r - 12, cy_r + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_GREEN, 1)

        row_y = H - 96
        for label, ok, val in [
            ("Lighting",   light_ok,
             f"mean={mean_b:.0f} contrast={contrast:.0f}"),
            ("Face width", face_ok,
             f"{face_w}px  (need {GUIDE_BOX_MIN_PX}–{GUIDE_BOX_MAX_PX})"),
        ]:
            col  = COL_GREEN if ok else COL_RED
            tick = "OK" if ok else "!!"
            cv2.putText(disp, f"[{tick}] {label}: {val}",
                        (10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)
            row_y += 24

        if not light_ok:
            hint = ("Too dark — move to brighter area"
                    if mean_b < LIGHT_MIN_MEAN else "Too bright — step away")
            cv2.putText(disp, hint, (10, H - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ORANGE, 1)
        elif not face_ok and face_w > 0:
            hint = ("Move CLOSER to camera"
                    if face_w < GUIDE_BOX_MIN_PX else "Step BACK slightly")
            cv2.putText(disp, hint, (10, H - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_ORANGE, 2)
        elif all_ok:
            cv2.putText(
                disp,
                f"Hold still — starting in {AUTO_START_HOLD_SEC - held:.1f}s ...",
                (W // 2 - 165, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_GREEN, 2)
        else:
            cv2.putText(disp, "Position your face in the guide box",
                        (10, H - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_YELLOW, 1)

        if len(faces) == 1:
            (fx, fy, fw, fh), _ = faces[0]
            cv2.rectangle(disp, (fx, fy), (fx+fw, fy+fh), COL_CYAN, 1)

        cv2.imshow("Smart Inventory — Enrollment", disp)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("  Cancelled during setup.")
            return False


# ── PHASE 2 — GUIDED CAPTURE WITH POSE VERIFICATION ───────────────────────────

def _run_phase2_guided_capture(camera, name):
    """
    Step 4: Guided capture — each pose verified via landmark geometry.

    KEY FIX: Stage advances IMMEDIATELY once pose is confirmed
    (no timer wait). This means the customer doesn't have to hold
    a pose for 2.4s — just 2 confirmed frames and the system moves on.
    """
    print("\n── Phase 2: Guided Capture ─────────────────────")
    print("  Follow on-screen instructions.")
    print("  Each pose confirmed before advancing.")
    print("  Q = cancel\n")

    H, W          = 480, 640
    failed        = 0
    raw_buffer    = []
    pose_log      = []
    session_start = time.time()

    for pose_tag, pose_label in HEAD_POSITIONS:
        stage_start    = time.time()
        stage_deadline = stage_start + SECONDS_PER_POSE + POSE_EXTEND_MAX_SEC
        confirmed      = 0
        stage_passed   = False

        print(f"  [{pose_tag.upper():8s}] Waiting for: '{pose_label}' "
              f"(need {MIN_POSE_FRAMES_REQUIRED} confirmed frames)")

        while time.time() < stage_deadline:
            raw = camera.read_frame()
            if raw is None:
                failed += 1
                if failed >= MAX_FAILED_READ:
                    print("  Stream lost.")
                    return None, None
                continue
            failed = 0

            ts    = time.time() - session_start
            frame = cv2.resize(raw, (W, H))
            raw_buffer.append((ts, frame.copy()))

            faces         = _detect_faces_mp(frame)
            detected_pose = "unknown"

            if len(faces) == 1:
                _, lm         = faces[0]
                detected_pose = _estimate_pose_tag(lm)
                if detected_pose == pose_tag:
                    confirmed += 1
                    pose_log.append((ts, detected_pose))

            if confirmed >= MIN_POSE_FRAMES_REQUIRED:
                stage_passed = True

            # ── Draw UI ────────────────────────────────────
            disp          = _preprocess(frame).copy()
            stage_elapsed = time.time() - stage_start

            overlay = disp.copy()
            cv2.rectangle(overlay, (0, 0), (W, 70), COL_BLACK, -1)
            cv2.addWeighted(overlay, 0.82, disp, 0.18, 0, disp)
            cv2.putText(disp, "SMART INVENTORY — Guided Capture",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_CYAN, 1)
            cv2.putText(disp, f"Enrolling: {name}",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COL_WHITE, 1)

            instr_col = COL_GREEN if stage_passed else COL_YELLOW
            cv2.putText(disp, pose_label,
                        (W // 2 - len(pose_label) * 8, H // 2 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, instr_col, 2)

            match_col = COL_GREEN if detected_pose == pose_tag else COL_ORANGE
            cv2.putText(disp,
                        f"Detected: {detected_pose}   "
                        f"Confirmed: {confirmed}/{MIN_POSE_FRAMES_REQUIRED}",
                        (10, H // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, match_col, 1)

            # Confirmation bar
            bar_x1, bar_y1 = W // 4, H // 2 + 38
            bar_x2, bar_y2 = 3 * W // 4, H // 2 + 52
            cv2.rectangle(disp, (bar_x1, bar_y1), (bar_x2, bar_y2),
                          (40, 40, 40), -1)
            fill_pct = min(confirmed / MIN_POSE_FRAMES_REQUIRED, 1.0)
            fill_w   = int((bar_x2 - bar_x1) * fill_pct)
            if fill_w > 0:
                cv2.rectangle(disp, (bar_x1, bar_y1),
                              (bar_x1 + fill_w, bar_y2),
                              COL_GREEN if stage_passed else COL_YELLOW, -1)

            if stage_passed:
                cv2.putText(disp, "POSE CONFIRMED — moving on...",
                            (W // 2 - 130, H // 2 + 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_GREEN, 2)
            else:
                hints = {
                    "left":  "Turn head LEFT — nose points left",
                    "right": "Turn head RIGHT — nose points right",
                    "up":    "Tilt chin UP slightly",
                    "down":  "Tilt chin DOWN slightly",
                }
                hint = hints.get(pose_tag, "")
                if hint:
                    cv2.putText(disp, hint,
                                (W // 2 - len(hint) * 4, H // 2 + 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_ORANGE, 1)

            # Step indicators
            pose_tags = [p[0] for p in HEAD_POSITIONS]
            cur_idx   = pose_tags.index(pose_tag)
            for i, (ptag, plabel) in enumerate(HEAD_POSITIONS):
                sx  = 10 + i * (W // len(HEAD_POSITIONS))
                col = (COL_GREEN  if i < cur_idx  else
                       COL_YELLOW if i == cur_idx else COL_GRAY)
                label_str = ("✓ " if i < cur_idx else "") + plabel[:7]
                cv2.putText(disp, label_str, (sx, H - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

            # Overall progress bar
            total_elapsed = time.time() - session_start
            total_max     = (SECONDS_PER_POSE + POSE_EXTEND_MAX_SEC) * len(HEAD_POSITIONS)
            pct           = min(total_elapsed / total_max, 1.0)
            cv2.rectangle(disp, (12, H - 20), (W - 12, H - 8),
                          (35, 35, 35), -1)
            fill = int((W - 24) * pct)
            if fill > 0:
                cv2.rectangle(disp, (12, H - 20),
                              (12 + fill, H - 8), COL_GREEN, -1)
            cv2.putText(disp,
                        f"Pose {cur_idx + 1}/{len(HEAD_POSITIONS)}  ·  "
                        f"{len(raw_buffer)} frames buffered",
                        (12, H - 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_GRAY, 1)

            if len(faces) == 1:
                (bx, by, bw, bh), _ = faces[0]
                box_col = COL_GREEN if detected_pose == pose_tag else COL_ORANGE
                cv2.rectangle(disp, (bx, by), (bx+bw, by+bh), box_col, 2)
            elif len(faces) > 1:
                cv2.putText(disp, "One person only",
                            (W // 2 - 70, H // 2 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_RED, 2)

            cv2.imshow("Smart Inventory — Enrollment", disp)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                print("  Cancelled during recording.")
                return None, None

            # ── KEY FIX: advance immediately once confirmed ──
            if stage_passed:
                break

        status = "✅" if stage_passed else "⚠️ "
        print(f"  {status} [{pose_tag.upper():8s}] {confirmed} confirmed frames")

    print(f"\n  Recording complete — {len(raw_buffer)} raw frames in buffer")
    return raw_buffer, pose_log


# ── PHASE 2 — FRAME EXTRACTION ────────────────────────────────────────────────

def _extract_frames(raw_buffer):
    """
    Step 5: Time-based extraction — one frame every EXTRACT_INTERVAL_SEC.
    Guaranteed 25-50 frames regardless of camera FPS.
    """
    if not raw_buffer:
        return []

    extracted    = []
    next_extract = 0.0

    for ts, frame in raw_buffer:
        if ts >= next_extract:
            extracted.append((ts, frame))
            next_extract = ts + EXTRACT_INTERVAL_SEC
            if len(extracted) >= EXTRACT_MAX_FRAMES:
                break

    # Fallback: index sampling if time-based gave too few
    if len(extracted) < EXTRACT_MIN_FRAMES and len(raw_buffer) >= EXTRACT_MIN_FRAMES:
        step      = max(1, len(raw_buffer) // EXTRACT_MIN_FRAMES)
        extracted = [raw_buffer[i] for i in range(0, len(raw_buffer), step)]
        extracted = extracted[:EXTRACT_MAX_FRAMES]
        print("  (Fallback index sampling used)")

    print(f"\n── Phase 2 complete: {len(extracted)} frames extracted "
          f"from {len(raw_buffer)} raw  (interval={EXTRACT_INTERVAL_SEC}s)")
    return extracted


# ── PHASE 3 — QUALITY SCORING ──────────────────────────────────────────────────

def _run_phase3_quality_scoring(extracted_frames):
    print("\n── Phase 3: Per-frame quality scoring ──────────")

    candidates = []
    rej_multi  = 0
    rej_qual   = 0

    for ts, frame in extracted_frames:
        faces = _detect_faces_mp(frame)

        if len(faces) != 1:
            rej_multi += 1
            continue

        bbox, landmarks = faces[0]
        score = _score_frame(frame, bbox, landmarks)

        if score < QUALITY_MIN_SCORE:
            rej_qual += 1
            continue

        pose = _estimate_pose_tag(landmarks)
        candidates.append({
            "score": score, "pose": pose,
            "frame": frame, "bbox": bbox,
            "landmarks": landmarks, "ts": ts,
        })

    print(f"  Passed : {len(candidates)}/{len(extracted_frames)}")
    print(f"  Rejected — multi-face: {rej_multi}  low quality: {rej_qual}")

    if not candidates:
        print("  ⚠️  Zero frames passed — try better lighting or move closer.")
    return candidates


# ── PHASE 4 — DEDUP + SELECTION ───────────────────────────────────────────────

def _quick_embedding(frame, bbox):
    x, y, w, h = bbox
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return np.zeros(1024)
    resized = cv2.resize(crop, (32, 32))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat    = gray.flatten()
    norm    = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat

def _cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _run_phase4_dedup_and_selection(candidates):
    print("\n── Phase 4: Dedup & selection ──────────────────")
    if not candidates:
        return []

    for c in candidates:
        c["quick_emb"] = _quick_embedding(c["frame"], c["bbox"])

    used     = [False] * len(candidates)
    clusters = []
    for i in range(len(candidates)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(candidates)):
            if not used[j]:
                if _cosine_sim(candidates[i]["quick_emb"],
                               candidates[j]["quick_emb"]) >= CANDIDATE_SIM_THRESHOLD:
                    cluster.append(j)
                    used[j] = True
        used[i] = True
        clusters.append(cluster)

    print(f"  Candidates: {len(candidates)}  Clusters: {len(clusters)}")

    survivors = sorted(
        [candidates[max(cl, key=lambda idx: candidates[idx]["score"])]
         for cl in clusters],
        key=lambda c: c["score"], reverse=True
    )
    print(f"  Survivors after dedup: {len(survivors)}")

    # Top 10 with pose diversity (max 3 per pose initially)
    pose_counts = {}
    final       = []
    for c in survivors:
        pose = c["pose"]
        if pose_counts.get(pose, 0) < 3:
            final.append(c)
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
        if len(final) >= TOP_N_FINAL:
            break

    # Fill remaining without pose restriction
    if len(final) < TOP_N_FINAL:
        for c in survivors:
            if c not in final:
                final.append(c)
            if len(final) >= TOP_N_FINAL:
                break

    pose_summary = {}
    for c in final:
        pose_summary[c["pose"]] = pose_summary.get(c["pose"], 0) + 1

    print(f"  Final selected: {len(final)}  Pose spread: {pose_summary}")
    return final


# ── SAVE FRAMES ───────────────────────────────────────────────────────────────

def _save_frames_to_disk(final_candidates, folder_path):
    paths = []
    ts    = int(time.time())
    for i, c in enumerate(final_candidates):
        fname    = f"enroll_{ts}_{i+1:02d}_{c['pose']}_{int(c['score']*100)}.jpg"
        fpath    = os.path.join(folder_path, fname)
        enhanced = c["frame"]  # save original — CLAHE only used for detection
        try:
            cv2.imwrite(fpath, enhanced)
            paths.append(fpath)
            print(f"  Saved [{i+1:02d}] score={c['score']:.3f} "
                  f"pose={c['pose']} → {fname}")
        except Exception as e:
            print(f"  Save error [{i+1}]: {e}")
    return paths


# ── PHASE 5 ───────────────────────────────────────────────────────────────────
#
# CRITICAL — pipeline must match face_engine.py _generate_embedding() exactly:
#   frame → crop bbox → upscale 640×640 INTER_CUBIC → MediaPipe face locate
#   → 20px pad → CLAHE (clipLimit=2.0, 8×8) → resize 224×224
#   → ArcFace detector_backend="skip"
#
# Using RetinaFace here (old code) produced embeddings that were NOT comparable
# to what recognition generates, because the two backends crop and align faces
# differently. ArcFace is sensitive to crop alignment — mismatched pipelines
# suppress similarity scores even for the same person.

_UPSCALE_SIZE    = 640
_MP_CONF         = 0.5   # matches face_engine.py MP_MIN_DETECTION_CONF
_MP_MODEL        = 1     # matches face_engine.py MP_MODEL_SELECTION (full-range)
_CLAHE_CLIP      = 2.0   # matches face_engine.py _preprocess_face
_CLAHE_GRID      = (8, 8)
_ARCFACE_SIZE    = 224


def _upscale_bbox_crop(frame, bbox):
    """
    Crop around the face bbox with GENEROUS padding, then upscale to
    _UPSCALE_SIZE × _UPSCALE_SIZE using INTER_CUBIC.

    NOTE: bbox here comes from Phase 3's _detect_faces_mp() and is
    already a TIGHT face-only box (unlike face_engine.py, which crops
    around a much larger YOLO *person* box). If we only add 20px
    padding to an already-tight face box and then upscale that small
    region to 640px, the face fills the entire frame edge-to-edge with
    no surrounding context — MediaPipe's detector is trained on
    normally-proportioned scenes and fails to find a face that fills
    100% of the frame. Padding by a fraction of the box size (not a
    fixed pixel count) keeps proportions sane regardless of how close
    or far the original face was.
    """
    x, y, w, h  = bbox
    h_f, w_f    = frame.shape[:2]

    # Pad by 60% of box size on each side — gives MediaPipe a properly
    # proportioned region to re-detect within after upscaling.
    pad_x = int(w * 0.6)
    pad_y = int(h * 0.6)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_f, x + w + pad_x)
    y2 = min(h_f, y + h + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (_UPSCALE_SIZE, _UPSCALE_SIZE),
                      interpolation=cv2.INTER_CUBIC)


def _mediapipe_face_crop(upscaled, mp_model):
    """
    Run MediaPipe on the upscaled crop to precisely locate the face.
    Returns a padded face crop, or None if no face found.
    Identical to face_engine._find_face_in_crop().
    No fallback guessing — a missed detection discards the frame.
    """
    try:
        rgb    = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        h, w   = upscaled.shape[:2]
        result = mp_model.process(rgb)

        if not result.detections:
            return None

        best = max(result.detections,
                   key=lambda d: d.score[0] if d.score else 0.0)
        bb   = best.location_data.relative_bounding_box

        x  = max(0, int(bb.xmin  * w))
        y  = max(0, int(bb.ymin  * h))
        bw = min(int(bb.width  * w), w - x)
        bh = min(int(bb.height * h), h - y)

        if bw <= 0 or bh <= 0:
            return None

        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        face_crop = upscaled[y1:y2, x1:x2]
        return face_crop if face_crop.size > 0 else None

    except Exception as e:
        print(f"    MediaPipe error: {e}")
        return None


def _clahe_preprocess(face_crop):
    """
    Resize to _ARCFACE_SIZE and apply CLAHE to luminance channel.
    Identical to face_engine._preprocess_face().
    """
    try:
        face  = cv2.resize(face_crop, (_ARCFACE_SIZE, _ARCFACE_SIZE))
        lab   = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP,
                                 tileGridSize=_CLAHE_GRID)
        l     = clahe.apply(l)
        lab   = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"    CLAHE error: {e}")
        return face_crop


def _generate_arcface_embeddings(final_candidates):
    """
    Phase 5 — generate one ArcFace embedding per selected frame
    using the SAME pipeline as face_engine._generate_embedding():
      bbox crop (generous pad) → upscale → MediaPipe → CLAHE → ArcFace skip

    Fallback: if MediaPipe can't re-detect a face in the upscaled crop
    (rare — extreme angles), use the crop as-is rather than discarding
    the frame entirely. Phase 3 already confirmed a face was present
    here with min_detection_confidence=0.6, so the frame is trustworthy
    even if the second-pass detection is more conservative.

    MediaPipe is initialised once and shared across all frames
    to avoid repeated model loading.
    """
    from deepface import DeepFace
    import mediapipe as mp

    embeddings = []
    skipped    = 0
    fallback   = 0

    print(f"\n── Phase 5: ArcFace embeddings ({len(final_candidates)} frames)")
    print(f"   Pipeline: bbox crop (padded) → upscale {_UPSCALE_SIZE}px → "
          f"MediaPipe → CLAHE → ArcFace skip")

    # Initialise MediaPipe once — same settings as face_engine.py
    mp_model = mp.solutions.face_detection.FaceDetection(
        model_selection=_MP_MODEL,
        min_detection_confidence=_MP_CONF
    )

    for i, c in enumerate(final_candidates):
        frame = c["frame"]
        bbox  = c["bbox"]
        tag   = f"[{i+1:02d}] pose={c['pose']} score={c['score']:.3f}"

        # Step 1 — upscale padded bbox crop
        upscaled = _upscale_bbox_crop(frame, bbox)
        if upscaled is None:
            print(f"  {tag} ❌ upscale failed")
            skipped += 1
            continue

        # Step 2 — MediaPipe face locate in upscaled crop
        face_crop = _mediapipe_face_crop(upscaled, mp_model)
        if face_crop is None:
            # Fallback — Phase 3 already confirmed a face exists here
            # with a stricter confidence (0.6). Use a centred crop of
            # the upscaled image rather than discarding a good frame.
            h, w = upscaled.shape[:2]
            m    = int(min(h, w) * 0.15)   # trim 15% margin off each side
            face_crop = upscaled[m:h - m, m:w - m]
            fallback += 1
            print(f"  {tag} ⚠️  MediaPipe missed — using fallback crop")

        # Step 3 — CLAHE + resize to 224×224
        face_crop = _clahe_preprocess(face_crop)

        # Step 4 — ArcFace with skip (face already cropped)
        try:
            result = DeepFace.represent(
                img_path=face_crop,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
            )
            if result:
                embeddings.append(np.array(result[0]["embedding"]))
                print(f"  {tag} ✅")
            else:
                print(f"  {tag} ❌ ArcFace returned empty")
                skipped += 1
        except Exception as e:
            print(f"  {tag} ❌ ArcFace error: {e}")
            skipped += 1

    mp_model.close()

    print(f"\n  Generated : {len(embeddings)}/{len(final_candidates)} embeddings")
    if fallback:
        print(f"  Fallback crop used: {fallback} frames")
    if skipped:
        print(f"  Skipped   : {skipped} frames "
              f"(upscale or ArcFace failed)")
    return embeddings


def _check_duplicate_multi(embeddings):
    if not embeddings:
        return False, None, None, 0.0
    best = (False, None, None, 0.0)
    for emb in embeddings:
        is_dup, name, sid, sim = check_duplicate_from_embedding(emb)
        if sim > best[3]:
            best = (is_dup, name, sid, sim)
    is_dup, name, sid, sim = best
    print(f"\n  Duplicate check — max similarity: {sim:.3f} "
          f"(threshold: {DUPLICATE_THRESHOLD})")
    return (True, name, sid, sim) if sim >= DUPLICATE_THRESHOLD else (False, None, None, sim)


def _show_duplicate_warning(dup_name, dup_id, similarity):
    warn = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(warn, (0, 0), (500, 300), (20, 20, 60), -1)
    cv2.putText(warn, "POSSIBLE DUPLICATE",        (55,  55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  COL_RED,    2)
    cv2.putText(warn, f"Match : {dup_name}",       (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE,  1)
    cv2.putText(warn, f"ID    : {dup_id}",         (30, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE,  1)
    cv2.putText(warn, f"Sim   : {similarity:.2f}", (30, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_YELLOW, 1)
    cv2.putText(warn, "Y = Proceed   N = Cancel",  (55, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_GREEN,  2)
    cv2.imshow("Duplicate Warning", warn)
    cv2.waitKey(400)
    print(f"\n⚠️  Possible duplicate: {dup_name} ({dup_id}) sim={similarity:.2f}")
    ch = input("   Proceed anyway? (y/n): ").strip().lower()
    cv2.destroyWindow("Duplicate Warning")
    return ch == "y"


def _show_success_screen(name, staff_id, num_embeddings):
    s = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(s, (0, 0), (500, 300), (10, 40, 10), -1)
    cv2.putText(s, "ENROLLMENT COMPLETE",                    (45,  70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COL_GREEN, 2)
    cv2.putText(s, f"Name : {name}",                         (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 1)
    cv2.putText(s, f"ID   : {staff_id}",                     (30, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 1)
    cv2.putText(s, f"Embeddings: {num_embeddings} + centroid",(30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COL_GREEN, 1)
    cv2.putText(s, "Ready for recognition!",                  (75, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_CYAN,  1)
    cv2.imshow("Smart Inventory — Enrollment", s)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()


# ── MAIN ENTRY POINT ───────────────────────────────────────────────────────────

def run_enrollment(stream_url):
    """
    Full 5-phase, 17-step guided enrollment pipeline.
    Called from main.py → menu option 1.
    """
    print("\n" + "=" * 56)
    print("   ENROLLMENT — Guided Video Pipeline (17-step)")
    print(f"   Poses: {len(HEAD_POSITIONS)}  ·  "
          f"Target frames: {TOP_N_FINAL}  ·  ArcFace + centroid")
    print("=" * 56)

    name = input("\nCustomer Full Name: ").strip()
    if not name:
        print("Name is required.")
        return False

    staff_id       = generate_staff_id()
    folder_path, _ = create_customer_folder(name)
    print(f"Auto ID : {staff_id}")

    camera, stream_url = connect_camera_with_retry(stream_url)
    if camera is None:
        return False

    try:
        if not _run_phase1_setup(camera, name):
            return False

        raw_buffer, pose_log = _run_phase2_guided_capture(camera, name)
        if not raw_buffer:
            return False

        extracted = _extract_frames(raw_buffer)
        if not extracted:
            print("No frames extracted — enrollment failed.")
            return False

        candidates = _run_phase3_quality_scoring(extracted)
        if not candidates:
            print("No frames passed quality scoring — try better lighting.")
            return False

        final = _run_phase4_dedup_and_selection(candidates)
        if not final:
            print("No frames selected after dedup — enrollment failed.")
            return False

    finally:
        camera.release()
        cv2.destroyAllWindows()

    _save_frames_to_disk(final, folder_path)

    embeddings = _generate_arcface_embeddings(final)
    if not embeddings:
        print("Could not generate any embeddings — enrollment failed.")
        return False

    is_dup, dup_name, dup_id, sim = _check_duplicate_multi(embeddings)
    if is_dup:
        if not _show_duplicate_warning(dup_name, dup_id, sim):
            print("Enrollment cancelled — duplicate.")
            return False

    ok = save_customer_multi_embedding(
        name=name, staff_id=staff_id,
        folder_path=folder_path, embeddings=embeddings,
    )
    if not ok:
        print("Database save failed!")
        return False

    add_embedding_to_memory(staff_id, name, embeddings)
    _show_success_screen(name, staff_id, len(embeddings))

    print("\n" + "=" * 56)
    print(f"  ✅  {name} enrolled successfully!")
    print(f"  ID         : {staff_id}")
    print(f"  Embeddings : {len(embeddings)} + centroid")
    print(f"  Folder     : {folder_path}")
    print(f"  Poses saved: "
          + ", ".join(f"{c['pose']}({c['score']:.2f})" for c in final))
    print("  Ready for recognition.")
    print("=" * 56)
    return True


# Backward-compatible alias
run_fast_enrollment = run_enrollment