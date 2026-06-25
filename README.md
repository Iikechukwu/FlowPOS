# FlowPOS

An AI-powered, checkout-free retail system that uses computer vision to identify customers by face and automatically tracks items they pick up from a shelf — billing them in real time with no manual checkout required.

---

## How It Works

1. **Customer walks in** — YOLOv8 detects a person at the entry point
2. **Face identified** — ArcFace matches the customer against enrolled identities within a 5-frame voting window
3. **Items tracked** — a second camera watches the shelf; YOLOv8 detects items taken or returned in real time
4. **Auto-billed** — when the customer leaves, the session closes and a receipt is written to the Excel log automatically

---

## Project Structure

```
FlowPOS/
│
├── main.py                  # Entry point — main menu
├── camera_stream.py         # Camera URL/index resolver
├── station_bridge.py        # Connects face recognition (M2) ↔ item detection (M4)
├── embedding_loader.py      # Loads face embeddings from DB into memory at startup
├── face_db.py               # SQLite interface — customers, embeddings, stats
├── admin_panel.py           # Admin UI — view customers, sales, logs
├── migrate_centroids.py     # One-time migration script for centroid format
├── test_integration.py      # Integration test suite
│
├── enrollment/
│   └── enrollment.py        # Customer face enrollment pipeline
│                            # MediaPipe detection → CLAHE → ArcFace embedding
│
├── m2/                      # Module 2 — Face Recognition & Access Control
│   ├── face_engine.py       # Core recognition engine (YOLOv8 + MediaPipe + ArcFace)
│   ├── access_controller.py # Runs recognition loop, fires CUSTOMER_IDENTIFIED events
│   └── events.py            # Event type definitions
│
├── m4/                      # Module 4 — Item Detection & Billing
│   ├── item_engine.py       # YOLOv8 shelf detection with tripwire logic
│   ├── billing.py           # Session billing, pricing, receipt generation
│   ├── excel_logger.py      # Writes transactions to Excel log
│   └── price_catalog.py     # Item name → price mapping
│
├── services/
│   ├── enrollment_manager.py  # Startup checks, system info
│   ├── embedding_service.py   # Standalone embedding generation utility
│   ├── duplicate_checker.py   # Prevents duplicate customer enrollment
│   └── face_quality.py        # Face image quality assessment
│
├── customers/               # Created at runtime — NOT committed to git
│   └── <name>/              # One folder per enrolled customer
│       └── enroll_*.jpg     # Enrollment face images
│
├── inventory.db             # SQLite database — NOT committed (biometric data)
├── yolov8n.pt               # YOLO model weights — NOT committed (download separately)
├── m4/best.pt               # Custom trained YOLO for shelf items — NOT committed
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/iikechukwu/FlowPOS.git
cd FlowPOS
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on OpenCV:** If you hit import conflicts, uninstall both packages and reinstall just one:
> ```bash
> pip uninstall opencv-python opencv-contrib-python
> pip install opencv-contrib-python==4.11.0.86
> ```

### 4. Download model weights

The YOLO weights are not included in this repo (too large for git). Download them manually:

| File | Purpose | Download |
|------|---------|----------|
| `yolov8n.pt` | Person detection (M2) | [Ultralytics releases](https://github.com/ultralytics/assets/releases) |
| `m4/best.pt` | Shelf item detection (M4) | Your custom trained model |

Place both files in the locations shown above before running.

### 5. Run the system

```bash
python main.py
```

On first run the database is created automatically. Use the main menu to enroll customers before starting recognition.

---

## Enrollment

From the main menu, select **Enroll Customer**. The enrollment pipeline:

- Uses **MediaPipe** to detect and crop the face from the camera frame
- Captures 10 frames across different angles (front, left, right, up, down)
- Applies **CLAHE** preprocessing to normalise lighting before embedding
- Runs each crop through **ArcFace** (`deepface`, `detector_backend="skip"`)
- Stores 10 embeddings + a precomputed centroid in the SQLite database

> Re-enroll any customer if recognition stops working — the stored centroid may have been polluted by outlier frames captured in poor lighting. The pairwise cosine similarity between all 10 stored embeddings should be ≥ 0.55; anything lower indicates a bad enrollment session.

---

## Key Design Decisions

**Why `detector_backend="skip"` in ArcFace?**
Face detection is already handled upstream by MediaPipe, which crops and aligns the face before ArcFace ever sees it. Passing the pre-cropped 224×224 image with `skip` avoids redundant detection and is significantly faster at runtime.

**Why a centroid instead of per-frame matching?**
Recognition compares the live embedding against a precomputed centroid (the average of all 10 enrollment embeddings). This handles natural intra-person variation across poses and lighting without needing to compare against every stored embedding on every frame.

**Why a voting buffer?**
A single blurry or partially occluded frame cannot trigger a false match. The system requires 3 agreeing votes out of a 5-frame buffer before confirming identity, making recognition stable under real-world camera conditions.

**Why StationBridge?**
M2 (face recognition) and M4 (item detection) run on separate camera threads with no shared state. `station_bridge.py` is the single source of truth for "who is currently active", decoupling the two modules completely so either can be developed or tested independently.

---

## Requirements

- Python 3.9+
- Webcam or RTSP stream (two cameras recommended — one for face, one for shelf)
- Windows 11 (primary tested platform; Linux support in progress)
- GPU optional but recommended for real-time YOLOv8 + ArcFace inference

See `requirements.txt` for the full pinned package list. Key dependencies:

| Package | Version | Role |
|---------|---------|------|
| deepface | 0.0.100 | ArcFace face embeddings |
| mediapipe | 0.10.9 | Real-time face detection & crop |
| ultralytics | — | YOLOv8 person & item detection |
| tensorflow | 2.19.0 | ArcFace model backend |
| tf_keras | 2.19.0 | Keras compatibility layer |
| opencv-contrib-python | 4.11.0.86 | Frame capture, CLAHE, Laplacian blur |
| torch | 2.4.1 | YOLOv8 inference backend |

---

## Known Limitations

These are the current boundaries of the system. Understanding them is important before deploying in a real environment.

**Single-person sessions only**
FlowPOS tracks one active customer at a time. If two people are in frame simultaneously, the system may fire conflicting `CUSTOMER_IDENTIFIED` and `CUSTOMER_LEFT` events, leading to billing errors. Multi-person tracking is the highest-priority improvement for v2.

**Lighting sensitivity**
The quality gate (Laplacian blur score ≥ 30, brightness 40–250) rejects frames that don't meet the threshold. In dim or unevenly lit environments, the system may silently drop all frames and never confirm a match. If recognition appears stuck, check the lighting conditions first.

**Enrollment quality dependency**
A corrupted enrollment session — one where some frames captured a bad angle, extreme pose, or poor lighting — will produce a centroid that is pulled in conflicting directions. The result is artificially low cosine similarity at match time even for the correct person. Re-enrollment is the fix; there is currently no automatic outlier detection during enrollment.

**Windows-only (current build)**
Camera index resolution and OpenCV bindings have only been tested on Windows 11. Linux and macOS support is planned but not yet verified.

**No liveness detection**
The system cannot distinguish a live face from a printed photograph or a screen displaying someone's face. A spoofing attack using a high-quality photo is theoretically possible. Depth-based or blink-detection liveness checks are not yet implemented.

**No offline-mode graceful degradation**
If the SQLite database is unavailable at startup, the system crashes rather than entering a safe fallback state.

---

## Planned Improvements

**Multi-person tracking**
Assign a persistent track ID to each YOLO person detection and maintain a separate recognition buffer and billing session per ID. This is the single most impactful change for real-world use.

**Adaptive quality gate**
At startup, sample 30 frames from the ambient environment and auto-calibrate the blur and brightness thresholds to the actual lighting conditions rather than using fixed constants.

**Liveness detection**
Integrate a blink detector or depth map check (using a stereo or structured-light camera) to prevent photo spoofing.

**Enrollment outlier filtering**
After generating the 10 enrollment embeddings, automatically discard any that fall more than one standard deviation below the mean pairwise similarity before computing the centroid. This prevents a single bad frame from corrupting the stored identity.

**Cloud sync & mobile receipts**
Replace the Excel logger with a lightweight API call that pushes the receipt to the customer's registered mobile number or email at session close, removing the Excel dependency entirely.

**Cross-platform packaging**
Containerise the system with Docker and abstract the camera index resolution so the same image runs on Linux edge devices (Raspberry Pi 5, NVIDIA Jetson) without code changes.

**Active learning enrollment**
After a successful recognition session, extract the best-quality frame from the live video and, if the cosine similarity is high enough, add it to the customer's embedding store to continuously improve the centroid over time.

---

## Important Notes

- **`customers/` and `inventory.db` are excluded from git** — they contain real face images and biometric embeddings. Never commit them to a public repository.
- **Model weights (`.pt` files) are excluded** — they exceed GitHub's file size limits. Download them separately using the links above.
- FlowPOS is designed for controlled indoor environments with consistent lighting. Recognition accuracy will degrade in outdoor settings or under variable or directional lighting.
- This is an academic/prototype system. It has not been audited for production security or GDPR/biometric data compliance.