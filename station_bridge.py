# station_bridge.py
#
# Integration glue between M2 (face recognition / access_controller.py)
# and M4 (billing.py / item_engine.py / excel_logger.py).
#
# Why this file exists:
#   M2's AccessController fires CUSTOMER_IDENTIFIED / CUSTOMER_LEFT events
#   from its own camera + thread. M4's ItemEngine watches a SEPARATE
#   camera pointed at the shelf and fires 'taken'/'returned' tripwire
#   events with no idea who's standing there. Neither side previously
#   had any code that connected the two — this file is that connection.
#
# Design:
#   - StationBridge owns the one piece of state both sides need to
#     share: "who is the currently active customer, if anyone."
#   - M2's camera/recognition thread calls handle_m2_event() every time
#     AccessController fires something (pass this as the on_event
#     callback when constructing AccessController).
#   - M3's camera runs in its own thread via run_item_detection_loop(),
#     calling ItemEngine.detect_with_tripwire() every frame and billing
#     whatever's currently active.
#   - A threading.Lock protects the active session, since both threads
#     touch it.

import threading
import time
import cv2

from m4.billing import LiveSession
from m4.excel_logger import log_item_event, log_session_summary
from m4.item_engine import ItemEngine
from camera_stream import CameraStream


class StationBridge:
    def __init__(self, item_model_path, item_camera_stream_url,
                 tripwire_line_x=420, item_conf=0.45):
        """
        item_model_path        : path to M3's trained YOLO weights (best.pt)
        item_camera_stream_url : stream URL for the SECOND camera,
                                  the one pointed at the shelf
        tripwire_line_x        : pixel x-coordinate of the tripwire —
                                  needs tuning to your actual camera framing
        """
        self._lock           = threading.Lock()
        self._active_session = None   # LiveSession or None

        # If the item-detection model fails to load (missing file,
        # corrupted .pt, wrong format), don't let that take down the
        # whole program. Face recognition (M2) should be able to run
        # independently even if billing/item-detection isn't ready —
        # better to run with billing disabled and a loud warning than
        # to crash before recognition even starts.
        self.item_engine = None
        try:
            self.item_engine = ItemEngine(
                model_path=item_model_path,
                line_x=tripwire_line_x,
                conf=item_conf
            )
            print("[BRIDGE] Item-detection model loaded successfully.")
        except Exception as e:
            print(f"⚠️  Item-detection model failed to load: {e}")
            print("⚠️  Billing/item-detection is DISABLED for this run. "
                  "Face recognition will still work normally.")

        self.item_camera_url = item_camera_stream_url
        self.item_camera     = None
        self.running         = True

    @property
    def item_detection_available(self):
        """True if the model loaded and item detection can actually run."""
        return self.item_engine is not None

    # ============================================
    # M2 SIDE — pass handle_m2_event as AccessController's on_event
    # ============================================
    def handle_m2_event(self, event: dict):
        """
        Called by AccessController every time it fires an event.
        event is the dict from build_event() in m2/events.py:
        {event, customer_id, name, confidence, num_faces}
        """
        event_type = event.get("event")

        if event_type == "CUSTOMER_IDENTIFIED":
            self._open_session(
                customer_id=event.get("customer_id"),
                name=event.get("name")
            )

        elif event_type == "CUSTOMER_LEFT":
            self._close_session()

        # NO_FACE / MULTIPLE_FACES — no billing action needed.
        # MULTIPLE_FACES could be surfaced to a UI layer later if
        # one gets built, but doesn't affect billing state itself.

    def _open_session(self, customer_id, name):
        with self._lock:
            if self._active_session is not None:
                # A session is already open — this shouldn't normally
                # happen given how access_controller.py guards state,
                # but if it does, don't silently overwrite an in-progress
                # bill. Log it loudly so it gets noticed during testing.
                print(f"⚠️  CUSTOMER_IDENTIFIED received while a session "
                      f"is already open for {self._active_session.customer_name} "
                      f"— ignoring new identification until the current "
                      f"session closes.")
                return

            self._active_session = LiveSession(name, customer_id)
            if self.item_engine is not None:
                self.item_engine.reset_session()
            print(f"[BRIDGE] Session opened — {name} ({customer_id})")

    def _close_session(self):
        with self._lock:
            if self._active_session is None:
                # CUSTOMER_LEFT with no open session — can happen if
                # M2 fires it during startup/edge cases. Harmless no-op.
                return

            summary = self._active_session.get_session_summary()
            log_session_summary(summary)
            print(f"[BRIDGE] Session closed — {summary['customer_name']} "
                  f"— total N{summary['total']}")

            self._active_session = None
            if self.item_engine is not None:
                self.item_engine.reset_session()

    # ============================================
    # M3/M4 SIDE — run this in its own thread
    # ============================================
    def run_item_detection_loop(self):
        """
        Owns the second camera. Runs continuously, feeding every
        tripwire crossing event to whichever session is currently
        active. If no session is active, detections are seen but
        not billed — nobody's there to bill.
        """
        if not self.item_detection_available:
            print("⚠️  Item detection loop not starting — model never "
                  "loaded successfully. Billing will not run, but face "
                  "recognition is unaffected.")
            return

        self.item_camera = CameraStream(stream_url=self.item_camera_url)
        if not self.item_camera.connect():
            print("⚠️  Item-detection camera failed to connect — "
                  "item billing will not run.")
            return

        print("[BRIDGE] Item detection loop started.")

        # NOTE: every frame is processed (no frame-skipping). An
        # earlier version of this loop skipped 2 out of every 3 frames
        # to avoid overwhelming the camera buffer, but that broke
        # ByteTrack's tracking continuity — persist=True tracking
        # assumes small frame-to-frame motion, and skipping frames
        # made fast-moving items (a hand picking something up) jump
        # far enough between detection calls that ByteTrack would
        # lose the track ID and reassign a new one, which meant
        # detect_with_tripwire()'s "tid in self._prev_x" check never
        # saw the object as already-tracked, so the crossing never
        # fired. imgsz=320 (set in ItemEngine) is the real fix for
        # keeping up with the camera in real time, proven in
        # standalone testing — full per-frame detection plus the
        # smaller inference resolution.
        while self.running:
            frame = self.item_camera.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            tripwire_events = self.item_engine.detect_with_tripwire(frame)

            if tripwire_events:
                with self._lock:
                    session = self._active_session

                if session is None:
                    # Items crossed the tripwire but nobody's
                    # identified right now — can't bill this to
                    # anyone. Log it so it's visible, don't crash,
                    # don't silently drop it either.
                    for e in tripwire_events:
                        print(f"⚠️  Item event with no active customer: "
                              f"{e['label']} ({e['direction']}) — not billed.")
                else:
                    for e in tripwire_events:
                        record = session.process_event(e)
                        log_item_event(record)

            # ── Live display window ──
            # item_engine.draw_overlay() now caches the YOLO result
            # from the most recent detect_with_tripwire() call on this
            # frame and reuses it, instead of re-running model.track()
            # — so this is a single inference pass per detection frame,
            # not two. It draws YOLO's actual boxes plus the tripwire
            # line and a live running bill, which is more informative
            # than a hand-rolled overlay would be.
            display = self.item_engine.draw_overlay(frame)

            with self._lock:
                active = self._active_session
            status_text = (
                f"Billing: {active.customer_name}"
                if active is not None else "No active customer"
            )
            cv2.putText(display, status_text, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Smart Inventory — Item Detection (Shelf Camera)",
                       display)

            # Q here stops ITEM DETECTION only — this loop runs on its
            # own thread, separate from AccessController's loop (which
            # owns the face camera and the real program shutdown). It
            # can't safely reach across threads to stop that loop too.
            # Press Q on the FACE camera window to stop the whole
            # program cleanly (already wired in access_controller.py).
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[BRIDGE] Q pressed on item detection window — "
                      "stopping item detection only.")
                print("[BRIDGE] Press Q on the FACE camera window to "
                      "stop the whole program.")
                break

        if self.item_camera:
            self.item_camera.release()
        try:
            cv2.destroyWindow(
                "Smart Inventory — Item Detection (Shelf Camera)"
            )
        except cv2.error:
            pass  # window already gone — harmless
        print("[BRIDGE] Item detection loop stopped.")

    def stop(self):
        self.running = False