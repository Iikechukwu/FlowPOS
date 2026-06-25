# item_engine.py
# Member 3 — Smart Inventory Surveillance System
# Snack detection + VERTICAL tripwire crossing tracker (bidirectional billing)
#
# Place best.pt in the same folder as this file, or pass the full path to ItemEngine().

from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2

from m4.price_catalog import PRICES as ITEM_PRICES
# Prices now come from price_catalog.py — the single source of truth.
# This file used to keep its own separate copy of prices, which drifted
# out of sync with price_catalog.py (different amounts for the same
# items, plus a "bottled_water" entry that isn't even one of the
# model's real trained classes). Importing avoids that happening again.
DEFAULT_PRICE = 0   # fallback for items not in the list


class ItemEngine:
    """
    Member 3 deliverable.
    YOLOv8 + ByteTrack with a VERTICAL tripwire for pickup tracking.

    Vertical line at pixel X = line_x.
    Centroid crosses LEFT  → RIGHT  →  'taken'    (charge +price)
    Centroid crosses RIGHT → LEFT   →  'returned' (credit  -price)

    Usage (now wired in automatically via station_bridge.py):
        engine = ItemEngine("m4/best.pt", line_x=420)
        events = engine.detect_with_tripwire(frame)   # call every frame
        bill   = engine.get_session_bill()             # call on session end
        engine.reset_session()                         # call before next customer
    """

    def __init__(self, model_path: str, line_x: int, conf: float = 0.45,
                 imgsz: int = 416):
        """
        imgsz: inference resolution. Lower = faster on CPU, at some
        accuracy cost. Must be a multiple of 32. 416 (down from
        YOLO's 640 default) cuts per-frame inference time meaningfully
        on non-GPU hardware — important since the live shelf camera
        loop has to keep up with the incoming stream in real time or
        risk dropped frames / disconnects (see station_bridge.py).
        """
        self.model  = YOLO(model_path)
        self.line_x = line_x
        self.conf   = conf
        self.imgsz  = imgsz
        self._prev_x:   dict[int, float] = {}
        self._log:      list[dict]       = []
        self._last_dir: dict[int, str]   = {}
        self._last_results = None   # cached YOLO output for the current
                                     # frame — lets draw_overlay() reuse
                                     # it instead of running inference twice

    # ------------------------------------------------------------------
    # PRIMARY — call every frame
    # ------------------------------------------------------------------
    def detect_with_tripwire(self, frame: np.ndarray) -> list[dict]:
        """
        Runs YOLO + ByteTrack. Returns crossing events this frame.
        Each event: {track_id, label, direction, price}
          direction='taken'    → price is positive  (item charged)
          direction='returned' → price is negative  (item credited)
        """
        results = self.model.track(
            frame, conf=self.conf, imgsz=self.imgsz, persist=True,
            tracker="bytetrack.yaml", verbose=False
        )
        self._last_results = results   # cache for draw_overlay()
        events = []
        if results[0].boxes.id is None:
            return events

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cid in zip(boxes, ids, classes):
            cx    = (box[0] + box[2]) / 2.0
            label = self.model.names[cid]
            if tid in self._prev_x:
                px = self._prev_x[tid]
                if   px < self.line_x <= cx:  direction = "taken"
                elif px > self.line_x >= cx:  direction = "returned"
                else:                         direction = None

                if direction and self._last_dir.get(tid) != direction:
                    unit_price   = ITEM_PRICES.get(label, DEFAULT_PRICE)
                    signed_price = unit_price if direction == "taken" else -unit_price
                    e = {
                        "track_id":  tid,
                        "label":     label,
                        "direction": direction,
                        "price":     signed_price,
                    }
                    events.append(e)
                    self._log.append(e)
                    self._last_dir[tid] = direction
            self._prev_x[tid] = cx
        return events

    # ------------------------------------------------------------------
    # SIMPLE DETECT — no tracking (optional helper)
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> dict[str, int]:
        """Simple per-frame detection without tripwire — returns {label: count}."""
        results = self.model(frame, conf=self.conf, verbose=False)
        counts: dict[str, int] = defaultdict(int)
        for box in results[0].boxes:
            counts[self.model.names[int(box.cls)]] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # SESSION END HELPERS
    # ------------------------------------------------------------------
    def get_session_items(self) -> dict[str, int]:
        """Net quantity per item (taken minus returned). Positive = still with customer."""
        tally: dict[str, int] = defaultdict(int)
        for e in self._log:
            tally[e["label"]] += 1 if e["direction"] == "taken" else -1
        return {k: v for k, v in tally.items() if v > 0}

    def get_session_bill(self) -> dict:
        """
        Returns full billing summary:
        {
          "items":       {label: {qty, unit_price, subtotal}},
          "total":       int,
          "event_count": int,
        }
        Returns already deducted for items returned.
        """
        items: dict = defaultdict(lambda: {"qty": 0, "unit_price": 0, "subtotal": 0})
        for e in self._log:
            label = e["label"]
            items[label]["unit_price"] = ITEM_PRICES.get(label, DEFAULT_PRICE)
            items[label]["qty"]      += 1 if e["direction"] == "taken" else -1
            items[label]["subtotal"] += e["price"]
        billed = {k: v for k, v in items.items() if v["qty"] > 0}
        total  = sum(v["subtotal"] for v in billed.values())
        return {"items": billed, "total": total, "event_count": len(self._log)}

    def reset_session(self):
        """Clear all state before next customer."""
        self._log.clear()
        self._prev_x.clear()
        self._last_dir.clear()

    # ------------------------------------------------------------------
    # DRAW OVERLAY — for display / UI
    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns annotated frame: boxes + vertical tripwire + live bill.

        Reuses the YOLO result already computed by the most recent
        detect_with_tripwire(frame) call on this same frame, instead
        of running inference a second time. Falls back to running its
        own inference only if called standalone, without
        detect_with_tripwire having run first on this frame.
        """
        if self._last_results is not None:
            results = self._last_results
        else:
            results = self.model.track(
                frame, conf=self.conf, imgsz=self.imgsz, persist=True,
                tracker="bytetrack.yaml", verbose=False
            )
        out = results[0].plot()
        h, w = out.shape[:2]

        # Tripwire line
        h, w = out.shape[:2]
        self.line_x = int(w * 0.60)
        cv2.line(out, (self.line_x, 0), (self.line_x, h), (0, 255, 255), 2)
        cv2.putText(out, "TRIPWIRE", (self.line_x + 6, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        mid_y = h // 2
        cv2.arrowedLine(out, (self.line_x - 40, mid_y), (self.line_x + 40, mid_y),
                        (0, 200, 255), 2, tipLength=0.3)
        cv2.putText(out, "taken", (self.line_x + 10, mid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Live bill (top-left)
        bill = self.get_session_bill()
        y = 55
        cv2.putText(out, "LIVE BILL:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for label, info in bill["items"].items():
            y += 25
            cv2.putText(out, f"  {label} x{info['qty']}  N{info['subtotal']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
        y += 30
        cv2.putText(out, f"TOTAL: N{bill['total']}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        return out