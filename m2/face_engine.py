# Pipeline (see recognition-pipeline diagram):
#   1. YOLO detects persons every frame (unchanged, 3-5m range)
#   2. If a session is already ACTIVE → follow that box
#      positionally (IoU match) — no ArcFace re-run per frame
#   3. If no session active → pick the largest box, ignore rest
#   4. Crop + upscale (INTER_CUBIC, unchanged)
#   5. MediaPipe finds the face in the crop (replaces Haar)
#   6. CLAHE preprocess the confirmed face crop
#   7. Buffer this frame's ArcFace embedding (skip detector)
#   8. After a short stability window, once 3 embeddings are
#      buffered: match each against the DB (centroid first,
#      individual-embedding fallback — via embedding_loader)
#   9. Majority vote — need 2 of 3 frames to agree on the same
#      person AND score >= threshold to become ACTIVE
#  10. Active session tracked by box IoU; 3s straight absence
#      ends the session and returns to step 3
# ============================================

import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import sys
import os
import time
import threading
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_loader import get_embedding_db, find_match, load_embeddings
from camera_stream import CameraStream, get_stream_url

# ============================================
# CONFIGURATION
# ============================================
RECOGNITION_THRESHOLD = 0.60   # match embedding_loader.DEFAULT_THRESHOLD
SHOW_WINDOW            = True
SESSION_END_SECONDS    = 1.5   # straight absence before session ends
                                # (lowered from 3.0 — shortens the window
                                # where a different person stepping into
                                # the same spot could be mistaken for the
                                # still-active customer via box-tracking)

# YOLO configuration
YOLO_MODEL              = "yolov8n.pt"
YOLO_CONFIDENCE         = 0.4
PERSON_CLASS            = 0

# Person-box significance filter (drop background/reflections)
MIN_PERSON_AREA_RATIO   = 0.08   # box must be >=8% of frame area
                                  # (raised from 0.05 — ignores people
                                  # passing by at a distance, only
                                  # triggers for someone actually at
                                  # the shelf)

# Stability before starting the embedding buffer
STABLE_SECONDS_REQUIRED = 0.8    # raised from 0.3 — avoids triggering
                                  # identification on someone briefly
                                  # walking past, not stopping at shelf

# Embedding buffer / majority vote
BUFFER_SIZE              = 5     # raised from 3 — more samples means
                                  # one bad frame (post quality-gate)
                                  # has less influence on the outcome
VOTES_REQUIRED           = 3     # raised from 2 — 3 of 5 agreeing is
                                  # more robust than 2 of 3, especially
                                  # now that weak frames are filtered
                                  # out before reaching the buffer

# Minimum average similarity across the winning votes. Without this,
# a candidate could win the majority vote with scores just barely
# over RECOGNITION_THRESHOLD (e.g. 0.61, 0.62, 0.60) — technically a
# match but not a confident one. This catches borderline cases that
# the vote count alone wouldn't.
VOTE_CONFIDENCE_FLOOR    = 0.65

# Active-session box tracking
IOU_MATCH_THRESHOLD      = 0.3   # min IoU to call it "the same person"

# Distance configuration
UPSCALE_SIZE            = 640

# MediaPipe configuration
MP_MIN_DETECTION_CONF   = 0.5
MP_MODEL_SELECTION      = 1      # full-range model — face may be far away

# Face crop quality gate — applied AFTER MediaPipe crop, BEFORE ArcFace.
# Rejects blurry or poorly-lit crops so weak frames never enter the
# vote buffer and drag down majority-vote confidence.
# Blur threshold is lower than face_quality.py's 100 because the crop
# at this point is a small region of the 640px upscaled frame —
# Laplacian variance naturally scores lower on smaller images.
QUALITY_MIN_BLUR        = 30     # Laplacian variance — below = too blurry
                                  # Lowered from 40: real-world testing showed
                                  # most frames scoring 30-40 were perfectly
                                  # usable for recognition but getting discarded,
                                  # making identification noticeably slow in
                                  # typical indoor lighting. 30 still blocks
                                  # genuinely blurry frames while letting
                                  # borderline-but-clear ones through.
QUALITY_MIN_BRIGHTNESS  = 40     # mean pixel value — below = too dark
QUALITY_MAX_BRIGHTNESS  = 250    # mean pixel value — above = too bright


# ============================================
# SESSION STATES
# ============================================
class SessionState:
    IDLE      = "IDLE"
    DETECTING = "DETECTING"
    ACTIVE    = "ACTIVE"
    ENDING    = "ENDING"


# ============================================
# RESULT BUILDER
# ============================================
def build_result(status, frame, num_faces=0,
                 customer_id=None, name=None,
                 confidence=None, box=None, face_crop=None):
    return {
        "status":      status,
        "num_faces":   num_faces,
        "customer_id": customer_id,
        "name":        name,
        "confidence":  confidence,
        "frame":       frame,
        "box":         box,         # (x, y, w, h) of the person this result
                                     # is about — populated for
                                     # UNKNOWN_CONFIRMED (cooldown/photo use)
        "face_crop":   face_crop    # raw face image (numpy array) — only
                                     # populated for UNKNOWN_CONFIRMED, the
                                     # photo to save for a temp UNK- record
    }


# ============================================
# BOX UTILITIES
# ============================================

def _iou(box_a, box_b):
    """
    Intersection-over-union for two (x, y, w, h) boxes.
    Used to follow the active session's person frame-to-frame
    without re-running recognition.
    """
    ax, ay, aw, ah = box_a[:4]
    bx, by, bw, bh = box_b[:4]

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union  = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _find_best_iou_match(target_box, candidate_boxes,
                         ambiguity_margin=0.10):
    """
    Find the candidate box with highest IoU against target_box.

    Returns (None, 0.0) if NO candidate has any real overlap with
    target_box (best_iou == 0.0) — this matters when the camera is
    pointed at an entirely different scene; without this guard the
    function would still hand back "the best of zero good options",
    which previously caused a tracked identity to be silently
    transferred onto an unrelated person.

    If a SECOND candidate is within `ambiguity_margin` of the best
    score (e.g. two people crossed paths and both now overlap the
    last known position similarly), tracking is unreliable — return
    no match at all rather than silently locking onto a possibly
    wrong person.

    Returns (best_box, best_iou) or (None, 0.0).
    """
    if not candidate_boxes:
        return None, 0.0

    scored = [(box, _iou(target_box, box)) for box in candidate_boxes]
    scored.sort(key=lambda s: s[1], reverse=True)

    best_box, best_iou = scored[0]

    if best_iou <= 0.0:
        return None, 0.0

    if len(scored) > 1:
        second_iou = scored[1][1]
        if (best_iou - second_iou) < ambiguity_margin:
            # Two boxes are too close to call — don't guess
            return None, 0.0

    return best_box, best_iou


# ============================================
# FACE ENGINE
# ============================================
class FaceEngine:

    def __init__(self, stream_url=None):
        self.stream_url = stream_url
        self.camera     = None
        self.connected  = False

        # ── Load YOLO ──
        print("Loading YOLO model...")
        try:
            self.yolo = YOLO(YOLO_MODEL)
            print("YOLO loaded! ✅")
        except Exception as e:
            print(f"YOLO load error: {e}")
            self.yolo = None

        # ── Load MediaPipe face detector (replaces Haar) ──
        print("Loading MediaPipe face detector...")
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=MP_MODEL_SELECTION,
                min_detection_confidence=MP_MIN_DETECTION_CONF
            )
            print("MediaPipe loaded! ✅")
        except Exception as e:
            print(f"MediaPipe load error: {e}")
            self._mp_face = None

        # ── Session state ──
        self.state                = SessionState.IDLE
        self.active_customer_id   = None
        self.active_customer_name = None
        self.active_confidence    = None
        self.active_box           = None     # last known box of active person
        self.absence_start        = None

        # ── Stability + buffering (pre-ACTIVE) ──
        self.candidate_box        = None      # box being evaluated
        self.candidate_seen_since = None      # timestamp first seen
        self.embedding_buffer     = []         # list of np arrays
        self.vote_buffer          = []         # list of match dicts/None
        self.face_crop_buffer     = []         # list of (face_crop, blur_score)
                                                # same index as vote_buffer —
                                                # lets us grab a real photo if
                                                # the vote confirms "unknown"

        # ── Threading ──
        self.latest_frame        = None
        self.frame_lock          = threading.Lock()
        self.recognition_running = False
        self.recognition_lock    = threading.Lock()
        self.current_persons     = []
        self.running             = True

        # ── Load embeddings ──
        print("Loading face database...")
        load_embeddings()
        self.embedding_db = get_embedding_db()
        print(f"Loaded {len(self.embedding_db)} customers ✅")

        # ── Connect camera ──
        self._connect_camera()

        # ── Start camera thread ──
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True
        )
        self.camera_thread.start()
        print("Camera thread started ✅")

    # ============================================
    # CAMERA
    # ============================================

    def _connect_camera(self):
        for attempt in range(3):
            try:
                print(f"Connecting camera ({attempt+1}/3)...")
                self.camera = CameraStream(stream_url=self.stream_url)
                if self.camera.connect():
                    self.connected = True
                    print("Camera connected! ✅")
                    return True
                print(f"Attempt {attempt+1} failed")
            except Exception as e:
                print(f"Error: {e}")
            if attempt < 2:
                time.sleep(2)

        self.connected = False
        print("⚠️  Camera failed!")
        return False

    def _camera_loop(self):
        failed = 0
        while self.running:
            if not self.connected or self.camera is None:
                time.sleep(0.1)
                continue

            frame = self.camera.read_frame()

            if frame is None:
                failed += 1
                if failed > 15:
                    print("Stream lost! Reconnecting...")
                    self.camera.release()
                    time.sleep(2)
                    self.camera = CameraStream(
                        stream_url=self.stream_url
                    )
                    if self.camera.connect():
                        failed = 0
                        print("Reconnected! ✅")
                continue

            failed = 0
            frame = cv2.resize(frame, (640, 480))

            with self.frame_lock:
                self.latest_frame = frame.copy()

    # ============================================
    # STAGE 1 — YOLO PERSON DETECTION
    # ============================================

    def _detect_persons_yolo(self, frame):
        """
        YOLO person detection — runs every frame (unchanged).
        Returns list of (x, y, w, h, conf) boxes, filtered to
        drop background/reflection-sized detections.
        """
        if self.yolo is None:
            return []

        try:
            results = self.yolo(
                frame,
                classes=[PERSON_CLASS],
                conf=YOLO_CONFIDENCE,
                verbose=False
            )

            persons = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    persons.append((x1, y1, x2-x1, y2-y1, conf))

            return self._filter_significant(persons, frame)

        except Exception as e:
            print(f"YOLO error: {e}")
            return []

    def _filter_significant(self, persons, frame):
        """
        Drop person boxes too small to be the primary subject —
        filters out background people / mirror reflections.
        """
        if not persons:
            return []

        h, w = frame.shape[:2]
        frame_area = h * w

        return [
            p for p in persons
            if (p[2] * p[3]) / frame_area >= MIN_PERSON_AREA_RATIO
        ]

    # ============================================
    # STAGE 2 — CROP + UPSCALE PERSON REGION
    # ============================================

    def _crop_and_upscale(self, frame, person_box):
        x, y, w, h = person_box[:4]
        h_f, w_f = frame.shape[:2]

        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_f, x + w + pad)
        y2 = min(h_f, y + h + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        upscaled = cv2.resize(
            crop,
            (UPSCALE_SIZE, UPSCALE_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        return upscaled

    # ============================================
    # STAGE 3 — FIND FACE IN UPSCALED CROP (MediaPipe)
    # ============================================

    def _find_face_in_crop(self, upscaled_crop):
        """
        Find face in the upscaled person crop using MediaPipe
        (full-range model — replaces Haar entirely).
        Returns face crop, or None if no face is genuinely found.
        No "guess top 40%" fallback — a missed detection should
        discard the frame, not feed a non-face into ArcFace.
        """
        if self._mp_face is None:
            return None

        try:
            rgb    = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2RGB)
            h, w   = upscaled_crop.shape[:2]
            result = self._mp_face.process(rgb)

            if not result.detections:
                return None

            # Use the most confident detection
            best = max(
                result.detections,
                key=lambda d: d.score[0] if d.score else 0.0
            )
            bb = best.location_data.relative_bounding_box

            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width  * w), w - x)
            bh = min(int(bb.height * h), h - y)

            if bw <= 0 or bh <= 0:
                return None

            # Add padding
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            face_crop = upscaled_crop[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            return face_crop

        except Exception as e:
            print(f"Face find error: {e}")
            return None

    def _preprocess_face(self, face_crop):
        try:
            face = cv2.resize(face_crop, (224, 224))

            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            return enhanced

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return face_crop

    # ============================================
    # STAGE 4a — FACE CROP QUALITY GATE
    # ============================================

    def _check_face_crop_quality(self, face_crop):
        """
        Lightweight blur + brightness check on the MediaPipe face crop,
        before CLAHE preprocessing and before ArcFace runs.

        Called after _find_face_in_crop() — the crop is already isolated
        so we're measuring actual face quality, not background noise.
        Called before _preprocess_face() — CLAHE changes brightness values
        so we must check the raw crop, not the enhanced one.

        Returns (passed, reason_string, blur_score).
        passed=False means this frame should be skipped — don't buffer it.
        blur_score is returned even on failure/error so callers that
        want to compare sharpness across frames don't need to
        recompute it themselves.
        """
        try:
            gray       = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = float(np.mean(gray))

            if blur_score < QUALITY_MIN_BLUR:
                return False, f"blurry ({blur_score:.1f} < {QUALITY_MIN_BLUR})", blur_score

            if brightness < QUALITY_MIN_BRIGHTNESS:
                return False, f"too dark ({brightness:.1f} < {QUALITY_MIN_BRIGHTNESS})", blur_score

            if brightness > QUALITY_MAX_BRIGHTNESS:
                return False, f"too bright ({brightness:.1f} > {QUALITY_MAX_BRIGHTNESS})", blur_score

            return True, f"ok (blur={blur_score:.1f} bright={brightness:.1f})", blur_score

        except Exception as e:
            # If quality check itself fails, let the frame through —
            # better to attempt recognition than to silently drop frames
            print(f"Quality check error: {e}")
            return True, "check failed — allowing frame", 0.0

    # ============================================
    # STAGE 4b — GENERATE ONE EMBEDDING (used per buffered frame)
    # ============================================

    def _generate_embedding(self, frame, person_box):
        """
        Run the full crop → upscale → MediaPipe → quality gate →
        preprocess → ArcFace pipeline for a single frame.

        Returns (embedding, face_crop, blur_score) on success.
        face_crop is the raw (pre-CLAHE) MediaPipe crop — kept around
        so a real photo is available if this frame ends up part of a
        confirmed "unknown person" vote.
        Returns (None, None, None) if any stage fails (including
        "no face genuinely found" or quality gate rejection).
        """
        upscaled = self._crop_and_upscale(frame, person_box)
        if upscaled is None:
            return None, None, None

        face_crop = self._find_face_in_crop(upscaled)
        if face_crop is None or face_crop.size == 0:
            return None, None, None

        # Quality gate — reject blurry or poorly-lit crops
        # before they can pollute the vote buffer
        passed, reason, blur_score = self._check_face_crop_quality(face_crop)
        if not passed:
            print(f"Frame rejected by quality gate: {reason}")
            return None, None, None

        preprocessed = self._preprocess_face(face_crop)

        try:
            result = DeepFace.represent(
                img_path=preprocessed,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )
            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"])
                return embedding, face_crop, blur_score
        except Exception as e:
            print(f"ArcFace error: {e}")

        return None, None, None

    # ============================================
    # BACKGROUND THREAD — ONE BUFFER SLOT AT A TIME
    # ============================================

    def _run_embedding_thread(self, frame, person_box):
        """
        Generates one embedding and appends it (plus its match
        result and face crop) to the buffers. Runs in the background
        so the camera/display loop never blocks on ArcFace.
        """
        try:
            emb, face_crop, blur_score = self._generate_embedding(
                frame, person_box
            )

            with self.recognition_lock:
                if emb is not None:
                    match = find_match(
                        emb, threshold=RECOGNITION_THRESHOLD
                    )
                    self.embedding_buffer.append(emb)
                    self.vote_buffer.append(match)
                    self.face_crop_buffer.append((face_crop, blur_score))
                # If emb is None, the frame is simply skipped —
                # buffer doesn't grow, we just try again next frame
                self.recognition_running = False

        except Exception as e:
            print(f"Embedding thread error: {e}")
            with self.recognition_lock:
                self.recognition_running = False

    # ============================================
    # MAJORITY VOTE
    # ============================================

    def _evaluate_votes(self):
        """
        Called once vote_buffer has BUFFER_SIZE entries.
        Returns the winning match dict if:
          1. >= VOTES_REQUIRED frames agree on the same staff_id
          2. the average similarity across those winning votes
             clears VOTE_CONFIDENCE_FLOOR (not just the per-frame
             RECOGNITION_THRESHOLD)
        Otherwise None (treated as UNKNOWN).
        """
        if len(self.vote_buffer) < BUFFER_SIZE:
            return None

        # Count votes by staff_id (None = no match that frame)
        ids = [m["staff_id"] for m in self.vote_buffer if m is not None]

        if not ids:
            return None

        counts = Counter(ids)
        winner_id, votes = counts.most_common(1)[0]

        if votes < VOTES_REQUIRED:
            return None

        # All winning-id matches among the buffered votes
        winning_matches = [
            m for m in self.vote_buffer
            if m is not None and m["staff_id"] == winner_id
        ]

        # Confidence floor — average similarity across winning votes
        # must clear VOTE_CONFIDENCE_FLOOR, not just the per-frame
        # threshold. Catches borderline cases like 3 votes that each
        # just barely passed RECOGNITION_THRESHOLD individually.
        avg_similarity = sum(m["similarity"] for m in winning_matches) / len(winning_matches)
        if avg_similarity < VOTE_CONFIDENCE_FLOOR:
            print(f"Vote won ({votes}/{len(self.vote_buffer)}) but average "
                  f"confidence {avg_similarity:.2f} below floor "
                  f"{VOTE_CONFIDENCE_FLOOR} — treating as UNKNOWN")
            return None

        # Use the highest-confidence match among the winning votes
        best = max(winning_matches, key=lambda m: m["similarity"])
        return best

    def _best_buffered_face_crop(self):
        """
        Returns the sharpest face crop currently sitting in
        face_crop_buffer (highest Laplacian blur score), or None
        if the buffer is empty. Used when a vote confirms a genuine
        unknown person, so we save a clean photo rather than
        whichever frame happened to be evaluated last.
        """
        if not self.face_crop_buffer:
            return None
        best_crop, best_score = max(
            self.face_crop_buffer, key=lambda c: c[1]
        )
        return best_crop

    def _reset_candidate(self):
        self.candidate_box        = None
        self.candidate_seen_since = None
        self.embedding_buffer     = []
        self.vote_buffer          = []
        self.face_crop_buffer     = []
        with self.recognition_lock:
            self.recognition_running = False

    # ============================================
    # DRAW UI
    # ============================================

    def _draw_ui(self, frame, persons):
        display = frame.copy()
        h, w    = display.shape[:2]

        # Identify which single box (if any) is the one actually
        # being tracked as the active/identified customer, so we
        # never label a second, different person with that name.
        # Uses the SAME threshold as process_frame's state logic —
        # a low/zero-overlap "best guess" must never count as a
        # real track, or the UI can label a stranger with the
        # active customer's name even after tracking has failed.
        tracked_box = None
        if self.state in (SessionState.ACTIVE, SessionState.ENDING) \
                and self.active_box is not None:
            candidate, score = _find_best_iou_match(self.active_box, persons)
            if candidate is not None and score >= IOU_MATCH_THRESHOLD:
                tracked_box = candidate

        for person in persons:
            x, y, pw, ph, conf = person
            is_tracked = (
                tracked_box is not None and
                person == tracked_box
            )

            if is_tracked and self.state == SessionState.ACTIVE:
                color = (0, 255, 0)
            elif is_tracked and self.state == SessionState.ENDING:
                color = (0, 165, 255)
            elif self.state == SessionState.DETECTING:
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(display, (x, y), (x+pw, y+ph), color, 2)

            if is_tracked and self.state in (SessionState.ACTIVE,
                                             SessionState.ENDING):
                conf_pct = self.active_confidence or 0
                label = f"{self.active_customer_name} ({conf_pct:.0%})"
            elif self.state in (SessionState.ACTIVE, SessionState.ENDING):
                # A session is occupied (or in its grace window) —
                # any other visible person is explicitly Unknown.
                # No recognition runs on them while this is true.
                label = "Unknown"
            elif self.state == SessionState.DETECTING:
                progress = len(self.vote_buffer)
                label = f"Identifying... ({progress}/{BUFFER_SIZE})"
            else:
                # IDLE, no session occupied — plain detection label
                label = f"Person ({conf:.0%})"

            cv2.rectangle(display, (x, y-28), (x+pw, y), color, -1)
            cv2.putText(display, label, (x+4, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        bar = display.copy()
        cv2.rectangle(bar, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.addWeighted(bar, 0.85, display, 0.15, 0, display)

        cv2.putText(display, "SMART INVENTORY", (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        state_colors = {
            SessionState.IDLE:      (100, 100, 100),
            SessionState.DETECTING: (0, 255, 255),
            SessionState.ACTIVE:    (0, 255, 0),
            SessionState.ENDING:    (0, 165, 255)
        }
        cv2.circle(display, (w-20, 20), 10,
                   state_colors.get(self.state, (100,100,100)), -1)

        if self.state == SessionState.IDLE:
            if persons:
                msg, color = "Person detected — identifying...", (0, 255, 255)
            else:
                msg, color = "Waiting for customer...", (100, 100, 100)

        elif self.state == SessionState.DETECTING:
            progress = len(self.vote_buffer)
            msg   = f"Please wait — identifying... ({progress}/{BUFFER_SIZE})"
            color = (0, 255, 255)

        elif self.state == SessionState.ACTIVE:
            conf  = self.active_confidence or 0
            msg   = f"Welcome, {self.active_customer_name}! ({conf:.0%})"
            color = (0, 255, 0)

        elif self.state == SessionState.ENDING:
            absent    = time.time() - self.absence_start
            remaining = max(0, SESSION_END_SECONDS - absent)
            msg       = f"Session ending in {remaining:.1f}s..."
            color     = (0, 165, 255)
        else:
            msg, color = "", (255, 255, 255)

        cv2.putText(display, msg, (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bot = display.copy()
        cv2.rectangle(bot, (0, h-28), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(bot, 0.7, display, 0.3, 0, display)
        cv2.putText(display, "Q = Quit | Smart Inventory System",
                   (10, h-9), cv2.FONT_HERSHEY_SIMPLEX,
                   0.42, (100, 100, 100), 1)

        return display

    # ============================================
    # MAIN PROCESS FRAME
    # ============================================

    def process_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return build_result("NO_FACE", None)
            frame = self.latest_frame.copy()

        # ── STAGE 1: YOLO person detection (every frame) ──
        persons = self._detect_persons_yolo(frame)
        num_persons = len(persons)
        self.current_persons = persons

        # ── STATE: ACTIVE — follow the same box via IoU ──
        if self.state == SessionState.ACTIVE:
            matched_box, iou_score = _find_best_iou_match(
                self.active_box, persons
            )

            if matched_box is not None and iou_score >= IOU_MATCH_THRESHOLD:
                self.active_box     = matched_box
                self.absence_start  = None
                result = build_result(
                    "STILL_PRESENT", frame, num_faces=num_persons,
                    customer_id=self.active_customer_id,
                    name=self.active_customer_name,
                    confidence=self.active_confidence
                )
            else:
                # Lost track of the active person this frame
                self.state         = SessionState.ENDING
                self.absence_start = time.time()
                result = build_result(
                    "STILL_PRESENT", frame, num_faces=0,
                    customer_id=self.active_customer_id,
                    name=self.active_customer_name
                )

        # ── STATE: ENDING — give them SESSION_END_SECONDS to return ──
        elif self.state == SessionState.ENDING:
            matched_box, iou_score = _find_best_iou_match(
                self.active_box, persons
            )

            if matched_box is not None and iou_score >= IOU_MATCH_THRESHOLD:
                self.state          = SessionState.ACTIVE
                self.active_box     = matched_box
                self.absence_start  = None
                result = build_result(
                    "STILL_PRESENT", frame, num_faces=num_persons,
                    customer_id=self.active_customer_id,
                    name=self.active_customer_name,
                    confidence=self.active_confidence
                )
            else:
                absent = time.time() - self.absence_start

                if absent >= SESSION_END_SECONDS:
                    ended_id   = self.active_customer_id
                    ended_name = self.active_customer_name

                    self.state                = SessionState.IDLE
                    self.active_customer_id   = None
                    self.active_customer_name = None
                    self.active_confidence    = None
                    self.active_box           = None
                    self.absence_start        = None
                    self._reset_candidate()

                    result = build_result(
                        "CUSTOMER_LEFT", frame,
                        customer_id=ended_id, name=ended_name
                    )
                else:
                    result = build_result(
                        "STILL_PRESENT", frame, num_faces=0,
                        customer_id=self.active_customer_id,
                        name=self.active_customer_name
                    )

        # ── STATE: IDLE or DETECTING — no active session yet ──
        else:
            if num_persons == 0:
                self._reset_candidate()
                self.state = SessionState.IDLE
                result = build_result("NO_FACE", frame)

            else:
                # Pick the largest box, ignore the rest entirely
                largest = max(persons, key=lambda p: p[2] * p[3])

                # Is this the same candidate we were already tracking?
                if self.candidate_box is not None:
                    same_person = _iou(self.candidate_box, largest) >= IOU_MATCH_THRESHOLD
                else:
                    same_person = False

                if not same_person:
                    # New candidate — reset stability + buffers
                    self._reset_candidate()
                    self.candidate_box        = largest
                    self.candidate_seen_since = time.time()
                    self.state = SessionState.IDLE
                    result = build_result("FACE_DETECTED", frame, num_faces=1)

                else:
                    self.candidate_box = largest
                    stable_for = time.time() - self.candidate_seen_since

                    if stable_for < STABLE_SECONDS_REQUIRED:
                        # Still warming up — not stable long enough yet
                        result = build_result("FACE_DETECTED", frame, num_faces=1)

                    else:
                        self.state = SessionState.DETECTING

                        # Kick off one embedding generation per call,
                        # as long as the buffer isn't full and no
                        # embedding job is currently running
                        with self.recognition_lock:
                            buffer_full = len(self.vote_buffer) >= BUFFER_SIZE
                            already_running = self.recognition_running

                            if not buffer_full and not already_running:
                                self.recognition_running = True
                                t = threading.Thread(
                                    target=self._run_embedding_thread,
                                    args=(frame, largest),
                                    daemon=True
                                )
                                t.start()

                        if len(self.vote_buffer) >= BUFFER_SIZE:
                            winner = self._evaluate_votes()

                            if winner:
                                self.state                = SessionState.ACTIVE
                                self.active_customer_id   = winner["staff_id"]
                                self.active_customer_name = winner["name"]
                                self.active_confidence    = winner["similarity"]
                                self.active_box           = largest
                                self.absence_start         = None
                                self._reset_candidate()

                                result = build_result(
                                    "IDENTIFIED", frame, num_faces=1,
                                    customer_id=winner["staff_id"],
                                    name=winner["name"],
                                    confidence=round(winner["similarity"], 2)
                                )
                            else:
                                # No majority — confirmed unknown person.
                                # Grab the sharpest buffered face crop as
                                # the photo to save before resetting —
                                # this is the same quality-gated frame
                                # data we already paid to generate, just
                                # not discarded this time.
                                best_crop = self._best_buffered_face_crop()
                                self._reset_candidate()

                                # Same as IDENTIFIED: start tracking this
                                # box as an active session right away.
                                # Without this, self.state would fall
                                # back to IDLE and the engine would
                                # immediately start a brand new 5-frame
                                # vote cycle on the very next frame —
                                # forever, for as long as this person
                                # stands at the shelf, even though a
                                # real session is already open one
                                # layer up in access_controller.py.
                                #
                                # The engine doesn't know the real
                                # UNK-<timestamp> staff_id yet — that's
                                # generated by access_controller.py a
                                # moment after this result is returned.
                                # That's fine: active_customer_id here
                                # is only ever used for the on-screen
                                # label in _draw_ui(), never read by
                                # the controller (it tracks the real
                                # id independently via its own state).
                                self.state                = SessionState.ACTIVE
                                self.active_customer_id   = None
                                self.active_customer_name = "Unknown"
                                self.active_confidence    = None
                                self.active_box           = largest
                                self.absence_start         = None

                                result = build_result(
                                    "UNKNOWN_CONFIRMED", frame, num_faces=1,
                                    box=largest,
                                    face_crop=best_crop
                                )
                        else:
                            result = build_result(
                                "DETECTING", frame, num_faces=1
                            )

        # ── Draw window ──
        if SHOW_WINDOW:
            display = self._draw_ui(frame, persons)
            if display is not None:
                cv2.imshow("Smart Inventory — Recognition", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return build_result("QUIT", frame)

        return result

    # ============================================
    # RESET + RELEASE
    # ============================================

    def reset_session(self):
        self.state                = SessionState.IDLE
        self.active_customer_id   = None
        self.active_customer_name = None
        self.active_confidence    = None
        self.active_box           = None
        self.absence_start        = None
        self._reset_candidate()
        print("Session reset!")

    def release(self):
        self.running = False
        time.sleep(0.5)
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Camera released!")