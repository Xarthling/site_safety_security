"""Zoning layer: per-camera zones, defined as polygons in config.yaml under `cameras:`.

  detection_zone  the camera's detection range. People and vehicles whose feet (the bottom centre of
                  their box) are outside it are ignored completely: no tracking, no rules, no events.
  gate_zone       (gate camera) vehicles entering or leaving this polygon produce vehicle_in and
                  vehicle_out events.

Polygons are written as fractions of the frame (0 to 1), so they stay correct at any resolution.
A camera with no entry under `cameras:` uses the whole frame. Use define_zone.py to draw a polygon
on a frame and get the numbers to paste into config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .types import Box

Point = Tuple[float, float]
Polygon = List[Point]


def anchor(box: Box) -> Point:
    """Where an object stands: the bottom centre of its box (feet / wheels on the ground)."""
    return (box[0] + box[2]) / 2.0, box[3]


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray casting: count how many polygon edges a ray going right from the point crosses."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass(frozen=True)
class ZoneSet:
    """One camera's zones in PIXELS for the clip being processed (None = not defined)."""
    detection: Optional[Polygon] = None
    gate: Optional[Polygon] = None

    def in_range(self, box: Box) -> bool:
        """Is this object inside the camera's detection range?"""
        return self.detection is None or point_in_polygon(*anchor(box), self.detection)


def _scale(points, width: int, height: int, camera: str, key: str) -> Optional[Polygon]:
    if not points:
        return None
    if len(points) < 3:
        raise ValueError(f"cameras.{camera}.{key} needs at least 3 points, got {len(points)}")
    scaled = []
    for p in points:
        if len(p) != 2 or not all(0.0 <= v <= 1.0 for v in p):
            raise ValueError(f"cameras.{camera}.{key}: every point must be [x, y] with values between 0 and 1, "
                             f"got {p}. (Zones are fractions of the frame; define_zone.py prints them.)")
        scaled.append((p[0] * width, p[1] * height))
    return scaled


def zones_for(cfg: dict, camera_id: str, width: int, height: int) -> ZoneSet:
    """Zones for one camera, scaled to the frame size. Cameras not listed use the whole frame."""
    spec = (cfg.get("cameras") or {}).get(camera_id) or {}
    return ZoneSet(_scale(spec.get("detection_zone"), width, height, camera_id, "detection_zone"),
                   _scale(spec.get("gate_zone"), width, height, camera_id, "gate_zone"))


def format_polygon(points_px: Sequence[Point], width: int, height: int) -> str:
    """Pixel points -> the YAML list to paste into config.yaml."""
    pairs = ", ".join(f"[{x / width:.3f}, {y / height:.3f}]" for x, y in points_px)
    return f"[{pairs}]"


@dataclass(frozen=True)
class ZoneEvent:
    kind: str          # "in" or "out"
    track_id: int
    time_s: float
    conf: float
    box: Box


class _State:
    def __init__(self):
        self.inside = False
        self.streak = 0            # consecutive frames that disagree with `inside`
        self.last_seen = 0.0
        self.box: Box = (0.0, 0.0, 0.0, 0.0)
        self.conf = 0.0


class InOutMonitor:
    """Reports when tracked objects enter and leave one polygon.

    A change only counts after it has held for hold_frames consecutive processed frames, so an object
    flickering on the edge does not create a stream of events. An object that disappears while inside
    is reported as leaving once it has been gone for lost_after_s. An object still inside when the clip
    ends gets no 'out' event."""

    def __init__(self, polygon: Polygon, hold_frames: int = 2, lost_after_s: float = 3.0):
        self.polygon, self.hold, self.lost_after_s = polygon, hold_frames, lost_after_s
        self.state: Dict[int, _State] = {}

    def update(self, tracks, t: float) -> List[ZoneEvent]:
        events: List[ZoneEvent] = []
        seen = set()
        for tr in tracks:
            st = self.state.setdefault(tr.track_id, _State())
            seen.add(tr.track_id)
            st.last_seen, st.box, st.conf = t, tr.box, tr.conf
            inside_now = point_in_polygon(*anchor(tr.box), self.polygon)
            if inside_now == st.inside:
                st.streak = 0
                continue
            st.streak += 1
            if st.streak >= self.hold:
                st.inside, st.streak = inside_now, 0
                events.append(ZoneEvent("in" if inside_now else "out", tr.track_id, t, tr.conf, tr.box))
        for tid, st in list(self.state.items()):
            if tid not in seen and t - st.last_seen >= self.lost_after_s:
                if st.inside:
                    events.append(ZoneEvent("out", tid, st.last_seen, st.conf, st.box))
                del self.state[tid]
        return events
