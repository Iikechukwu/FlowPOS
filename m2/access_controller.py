# ============================================
# SMART INVENTORY SYSTEM
# Module: Access Controller
# Author: Member 2
#
# Runs the main recognition loop.
# Reads face_engine results every frame.
# Fires events to M4 via callback.
#
# M2 is the ON/OFF switch for the system:
#   Nothing is logged without CUSTOMER_IDENTIFIED
#   Nothing closes without CUSTOMER_LEFT
#
# Design notes:
#   - FaceEngine owns ALL session state, absence
#     timing, and ENDING grace window internally.
#     AccessController does NOT duplicate any of
#     that logic — it only reacts to what the
#     engine reports.
#   - MULTIPLE_FACES warning fires once per
#     incident. Flag resets when resolved.
#   - UNKNOWN_CONFIRMED (majority vote found no
#     match, same confidence bar as IDENTIFIED)
#     immediately creates a temporary UNK- customer
#     record from the captured photo and opens a
#     session under that ID — same as a known
#     employee. A manager reviews/names them later
#     in admin_panel.py.
#   - end_session() is exposed for M4 to call
#     when the transaction is fully closed on
#     their side.
# ============================================

import time
from m2.face_engine import FaceEngine
from m2.events import Events, build_event
from face_db import create_unknown_customer


# ============================================
# CONFIGURATION
# ============================================

LOOP_DELAY = 0.03   # seconds between frames (~33fps cap)

CLEAN_FRAMES_TO_RESET = 5   # consecutive single-face frames required
                            # before the MULTIPLE_FACES warning flag
                            # resets — debounces flickery detections
                            # of a second person near the frame edge


# ============================================
# STATION STATES
# ============================================

class StationState:
    IDLE   = "IDLE"    # No active customer session
    ACTIVE = "ACTIVE"  # Known employee identified, session open


# ============================================
# ACCESS CONTROLLER
# ============================================

class AccessController:

    def __init__(self, on_event, stream_url=None):
        """
        on_event  : callback from M4 — receives every fired event dict
        stream_url: DroidCam URL, or None for laptop webcam
        """
        self.on_event = on_event
        self.engine   = FaceEngine(stream_url=stream_url)

        # Station state — IDLE until a customer is identified
        self.state = StationState.IDLE

        # Active customer info — set on IDENTIFIED, cleared on CUSTOMER_LEFT
        self.active_customer_id   = None
        self.active_customer_name = None
        self.active_confidence    = None

        # Multiple faces flag — prevents flooding M4 every frame
        # Resets when back to a normal single-person state
        self.multiple_faces_warned = False
        # Debounce counter — how many consecutive clean single-face
        # frames we've seen since the last MULTIPLE_FACES warning.
        # Must reach CLEAN_FRAMES_TO_RESET before the warning flag
        # resets, so a flickering second face (detected on/off frame
        # to frame near the edge) doesn't re-trigger the warning
        # repeatedly for what is really one ongoing incident.
        self.clean_face_count = 0

        print("Access Controller initialised.")
        print(f"Station state : {self.state}")

    # ============================================
    # INTERNAL HELPERS
    # ============================================

    def _fire(self, event_type, confidence=None, num_faces=0):
        """
        Build a standardised event payload and send it to M4.
        Always carries the active customer info when available.
        """
        event = build_event(
            event_type  = event_type,
            customer_id = self.active_customer_id,
            name        = self.active_customer_name,
            confidence  = confidence,
            num_faces   = num_faces
        )
        print(f"[EVENT] {event}")
        self.on_event(event)

    def _open_session(self, customer_id, name, confidence):
        """Transition IDLE → ACTIVE and store customer identity."""
        self.state                = StationState.ACTIVE
        self.active_customer_id   = customer_id
        self.active_customer_name = name
        self.active_confidence    = confidence
        self.multiple_faces_warned = False
        self.clean_face_count      = 0
        print(f"Session OPEN  — {name} ({customer_id})")

    def _close_session(self):
        """Transition ACTIVE → IDLE and clear all customer state."""
        print(f"Session CLOSED — {self.active_customer_name}")
        self.state                = StationState.IDLE
        self.active_customer_id   = None
        self.active_customer_name = None
        self.active_confidence    = None
        self.multiple_faces_warned = False
        self.clean_face_count      = 0
    # ============================================

    def handle_frame(self, result):
        """
        Called every loop iteration with the dict returned by
        face_engine.process_frame().

        The engine owns all timing (absence timer, ENDING grace
        window, stability window, vote buffer). This method only
        decides which events to fire based on what the engine reports.

        Status values the engine can return:
          NO_FACE           — no person in frame (IDLE/DETECTING)
          FACE_DETECTED     — person seen, stability window not passed yet
          DETECTING         — buffering embeddings, vote not ready
          IDENTIFIED        — majority vote confirmed a known person
          UNKNOWN_CONFIRMED — majority vote found no match (full confidence)
          STILL_PRESENT     — active session, person tracked (or in grace)
          CUSTOMER_LEFT     — engine's 3s absence timer expired, session reset
          QUIT              — user pressed Q
        """
        status    = result["status"]
        num_faces = result.get("num_faces", 0)

        # ── QUIT ────────────────────────────────────────────────
        if status == "QUIT":
            raise KeyboardInterrupt

        # ── NO_FACE ─────────────────────────────────────────────
        # Engine is IDLE or DETECTING, nobody in frame.
        # No action needed — engine handles absence internally.
        if status == "NO_FACE":
            return

        # ── FACE_DETECTED ────────────────────────────────────────
        # Person visible but stability window not yet passed.
        # Too early to do anything — just wait.
        if status == "FACE_DETECTED":
            return

        # ── DETECTING ────────────────────────────────────────────
        # Engine is buffering embeddings for vote.
        # Nothing to fire yet.
        if status == "DETECTING":
            return

        # ── IDENTIFIED ───────────────────────────────────────────
        # Majority vote confirmed a known employee.
        # Only opens a session if we're currently IDLE — prevents
        # re-firing if the engine somehow returns IDENTIFIED twice.
        if status == "IDENTIFIED":
            self.multiple_faces_warned = False
            self.clean_face_count      = 0

            if self.state == StationState.IDLE:
                customer_id = result["customer_id"]
                name        = result["name"]
                confidence  = result["confidence"]

                self._open_session(customer_id, name, confidence)
                self._fire(
                    Events.CUSTOMER_IDENTIFIED,
                    confidence=confidence,
                    num_faces=1
                )
            return

        # ── UNKNOWN_CONFIRMED ────────────────────────────────────
        # Majority vote ran and confirmed: no match, full confidence
        # (same reliability bar as IDENTIFIED — not a lighter check).
        # We don't just warn and wait for them to leave — we create
        # a temporary UNK- customer record from the photo, open a
        # real session under that ID, and tell M4 exactly as if this
        # were a known employee. M4 doesn't need to know the ID is
        # temporary; purchases get logged against it like anyone
        # else's, and a manager can review/assign a name later in
        # admin_panel.py.
        if status == "UNKNOWN_CONFIRMED":
            if self.state == StationState.IDLE:
                face_crop = result.get("face_crop")
                staff_id  = create_unknown_customer(face_crop)

                if staff_id is None:
                    # DB/disk failure — don't open a session we can't
                    # back with a real record. Stay IDLE and let the
                    # engine re-evaluate this person fresh next time.
                    print("Failed to create unknown customer record — "
                          "skipping session open.")
                    return

                self._open_session(staff_id, "Unknown", confidence=None)
                self._fire(
                    Events.CUSTOMER_IDENTIFIED,
                    confidence=None,
                    num_faces=1
                )
            return

        # ── STILL_PRESENT ────────────────────────────────────────
        # Active session — customer tracked (or in 3s grace window).
        # Reset the multiple-faces flag if we're back to normal.
        # No event fired — M4 only needs IDENTIFIED and CUSTOMER_LEFT.
        if status == "STILL_PRESENT":
            if self.state == StationState.ACTIVE:
                if num_faces == 1:
                    # Require a short streak of clean single-person
                    # frames before resetting the flag — num_faces is
                    # a raw per-frame YOLO count with no smoothing, so
                    # a second person near the frame edge can flicker
                    # between 1 and 2 from frame to frame. Resetting
                    # on a single clean frame let that flicker punch
                    # through the "only warn once" guard and re-fire
                    # MULTIPLE_FACES repeatedly for the same incident.
                    self.clean_face_count += 1
                    if self.clean_face_count >= CLEAN_FRAMES_TO_RESET:
                        self.multiple_faces_warned = False

                elif num_faces > 1:
                    # Multiple people at shelf during active session
                    self.clean_face_count = 0
                    if not self.multiple_faces_warned:
                        self._fire(
                            Events.MULTIPLE_FACES,
                            num_faces=num_faces
                        )
                        self.multiple_faces_warned = True
            return

        # ── CUSTOMER_LEFT ────────────────────────────────────────
        # Engine's absence timer expired (1.5s straight absence).
        # Engine has already reset its own internal state.
        # We close the station session and notify M4.
        if status == "CUSTOMER_LEFT":
            if self.state == StationState.ACTIVE:
                self._fire(Events.CUSTOMER_LEFT)
                self._close_session()
            return

    # ============================================
    # MAIN LOOP
    # ============================================

    def run(self):
        """
        Continuous loop — reads frames from the engine,
        passes each result to handle_frame().
        Runs until interrupted (Ctrl+C or Q key).
        """
        print("\nAccess Controller running...")
        print("Press Q in the camera window or Ctrl+C to stop.\n")

        try:
            while True:
                result = self.engine.process_frame()
                self.handle_frame(result)
                time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            self.engine.release()
            print("Access Controller stopped.")

    # ============================================
    # EXTERNAL CONTROL — called by M4
    # ============================================

    def end_session(self):
        """
        M4 calls this once the transaction is fully closed on
        their side (billing saved, receipt generated, etc.).
        Resets station to IDLE ready for the next customer.
        """
        if self.state == StationState.ACTIVE:
            print(f"M4 closed transaction for "
                  f"{self.active_customer_name}.")
            self._close_session()
            # Also tell engine to reset in case it's mid-ENDING
            self.engine.reset_session()
        else:
            print("end_session() called but no active session.")