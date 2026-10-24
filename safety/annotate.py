"""Draw detections, tracks and PPE status on a frame, for the annotated video and event snapshots."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .types import MISSING, OK, Det, Track
from .zones import ZoneSet

# BGR colours
COLOURS = {"person": (200, 160, 60), "vehicle": (40, 140, 240)}
GREEN, RED, DEFAULT_COLOUR, DET_COLOUR = (60, 180, 75), (40, 40, 220), (200, 200, 200), (0, 200, 230)
ZONE_COLOUR, GATE_COLOUR = (220, 200, 0), (220, 0, 220)

Statuses = Dict[int, Dict[str, Tuple[str, float]]]  # track_id -> item -> (status, confidence)


def blur_box(image: np.ndarray, box) -> np.ndarray:
    """Return a copy with the region inside box blurred (used to hide plates in saved snapshots)."""
    out = image.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (0, 0), 12)
    return out


def draw(image: np.ndarray, dets: Optional[List[Det]] = None, tracks: Optional[List[Track]] = None,
         caption: str = "", statuses: Optional[Statuses] = None,
         highlight_id: Optional[int] = None, vehicles: Optional[List[Track]] = None,
         vehicle_labels: Optional[Dict[int, str]] = None, zones: Optional[ZoneSet] = None,
         highlight_vehicle_id: Optional[int] = None) -> np.ndarray:
    """Return a copy of the image. Raw detections are thin yellow boxes; tracks are thick boxes
    with ID and PPE status (red = something missing, green = all ok); highlight_id is drawn thicker."""
    out = image.copy()
    if zones is not None:   # detection range: thin cyan outline; gate zone: thick magenta
        for polygon, colour, thick, name in ((zones.detection, ZONE_COLOUR, 1, "detection zone"),
                                             (zones.gate, GATE_COLOUR, 3, "gate zone")):
            if polygon:
                points = np.array(polygon, dtype=np.int32)
                cv2.polylines(out, [points], True, colour, thick)
                cv2.putText(out, name, (int(polygon[0][0]) + 4, int(polygon[0][1]) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    for d in dets or []:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), DET_COLOUR, 1)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (x1, max(10, y2 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, DET_COLOUR, 1)
    for t in tracks or []:
        status = (statuses or {}).get(t.track_id, {})
        values = [s for s, _ in status.values()]
        if MISSING in values:
            colour = RED
        elif values and all(v == OK for v in values):
            colour = GREEN
        else:
            colour = COLOURS.get(t.label, DEFAULT_COLOUR)
        x1, y1, x2, y2 = (int(v) for v in t.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 5 if t.track_id == highlight_id else 2)
        label = f"#{t.track_id} " + " ".join(f"{k}:{s}" for k, (s, _) in status.items())
        cv2.putText(out, label.strip(), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    for v in vehicles or []:   # vehicles have their own IDs, so they are drawn separately from people
        x1, y1, x2, y2 = (int(val) for val in v.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOURS["vehicle"], 6 if v.track_id == highlight_vehicle_id else 3)
        text = (vehicle_labels or {}).get(v.track_id, "vehicle")
        cv2.putText(out, f"V{v.track_id} {text}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOURS["vehicle"], 2)
    if caption:
        cv2.putText(out, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
