"""Gate camera: log every vehicle that passes, with its class and licence plate.

Per processed frame: detect vehicles (a general model, separate from the PPE model), give each a
stable ID (own Tracker), and, while a vehicle is large enough and its plate is not yet settled,
read plates from a crop around it. Class and plate are decided by voting over the vehicle's whole
pass, so one blurry frame cannot spoil the record. A vehicle is logged once, a few seconds after it
leaves view (or at the end of the clip, via flush).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .plates import PlateRead, plate_for_storage
from .tracking import Tracker
from .types import Box, Det, Track, VehicleRecord

CONFIRM_READS = 2   # stop reading once the leading plate has been read this many times
MAX_ATTEMPTS = 8    # ...or after this many attempts, so a bad view cannot cost compute forever
MIN_FRAMES = 3      # tracks shorter than this are ghosts and are not logged


@dataclass
class Finished:
    record: VehicleRecord
    crop: Optional[np.ndarray]      # best picture of the vehicle (the read that was most confident)
    plate_box: Optional[Box]        # where the plate is inside that crop, if one was read


@dataclass
class VehicleFrame:
    tracks: List[Track]             # vehicles currently tracked
    labels: Dict[int, str]          # track_id -> text to draw, e.g. "car ABC123"
    finished: List[Finished]        # vehicles that were logged on this frame


def iou(a: Box, b: Box) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_class(box: Box, dets: List[Det]) -> str:
    """Class of the detection that overlaps this tracked box the most."""
    best = max(dets, key=lambda d: iou(box, d.box), default=None)
    return best.cls if best is not None and iou(box, best.box) > 0 else "vehicle"


class _State:
    def __init__(self, t: float, box: Box):
        self.first_seen = self.last_seen = t
        self.box = box
        self.frames = 0
        self.cls_sum: Dict[str, float] = {}     # class -> summed confidence
        self.cls_n: Dict[str, int] = {}
        self.plate_score: Dict[str, float] = {}  # plate text -> summed confidence
        self.plate_hits: Dict[str, int] = {}
        self.plate_best: Dict[str, float] = {}   # plate text -> best single confidence
        self.attempts = 0
        self.plates_done = False
        self.best_read_conf = 0.0
        self.crop: Optional[np.ndarray] = None
        self.crop_area = 0.0
        self.plate_box: Optional[Box] = None

    def top_class(self) -> str:
        return max(self.cls_sum, key=self.cls_sum.get) if self.cls_sum else "vehicle"

    def top_plate(self) -> Optional[str]:
        return max(self.plate_score, key=self.plate_score.get) if self.plate_score else None


class VehicleMonitor:
    """State for ONE gate camera. The detector and plate reader are shared model objects."""

    def __init__(self, detector, reader, cfg: dict, sample_fps: float):
        """cfg is the `gate` section of config.yaml. reader may be None (classification only)."""
        self.detector, self.reader = detector, reader
        self.classes = set(cfg["vehicle_classes"])
        self.plate_min_conf = cfg["plate_min_conf"]
        self.min_width = cfg["min_vehicle_width_px"]
        self.finalize_after_s = cfg["finalize_after_s"]
        self.privacy, self.salt = cfg.get("store_plate", "text"), cfg.get("plate_salt", "")
        self.tracker = Tracker("vehicle", sample_fps)
        self.state: Dict[int, _State] = {}

    def process(self, image: np.ndarray, t: float) -> VehicleFrame:
        dets = [d for d in self.detector(image) if d.cls in self.classes]
        tracks = self.tracker.update(dets)
        for tr in tracks:
            st = self.state.get(tr.track_id)
            if st is None:
                st = self.state[tr.track_id] = _State(t, tr.box)
            st.last_seen, st.box, st.frames = t, tr.box, st.frames + 1
            cls = best_class(tr.box, dets)
            st.cls_sum[cls] = st.cls_sum.get(cls, 0.0) + tr.conf
            st.cls_n[cls] = st.cls_n.get(cls, 0) + 1

            width, area = tr.box[2] - tr.box[0], (tr.box[2] - tr.box[0]) * (tr.box[3] - tr.box[1])
            if self.reader is not None and not st.plates_done and width >= self.min_width:
                self._read_plate(st, image, tr.box)
            if st.best_read_conf == 0.0 and area > st.crop_area:   # no plate yet: keep the biggest view
                st.crop, st.crop_area = self._crop(image, tr.box), area
        labels = {tr.track_id: self._label(self.state[tr.track_id]) for tr in tracks}
        return VehicleFrame(tracks, labels, self._finalize(t, flush=False))

    def flush(self, t: float) -> List[Finished]:
        """Log every vehicle still in view (call at the end of a clip)."""
        return self._finalize(t, flush=True)

    # -- internals ------------------------------------------------------------------------

    @staticmethod
    def _crop(image: np.ndarray, box: Box) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
        return image[y1:y2, x1:x2].copy()

    def _read_plate(self, st: _State, image: np.ndarray, box: Box) -> None:
        crop = self._crop(image, box)
        st.attempts += 1
        reads: List[PlateRead] = self.reader(crop)
        for r in reads:
            if r.conf < self.plate_min_conf:
                continue
            st.plate_score[r.text] = st.plate_score.get(r.text, 0.0) + r.conf
            st.plate_hits[r.text] = st.plate_hits.get(r.text, 0) + 1
            st.plate_best[r.text] = max(st.plate_best.get(r.text, 0.0), r.conf)
            if r.conf > st.best_read_conf:
                st.best_read_conf, st.crop, st.plate_box = r.conf, crop, r.box
        top = st.top_plate()
        if st.attempts >= MAX_ATTEMPTS or (top and st.plate_hits[top] >= CONFIRM_READS):
            st.plates_done = True

    def _label(self, st: _State) -> str:
        plate = plate_for_storage(st.top_plate(), self.privacy, self.salt)
        return f"{st.top_class()} {plate}" if plate else st.top_class()

    def _finalize(self, t: float, flush: bool) -> List[Finished]:
        done = []
        for tid, st in list(self.state.items()):
            if flush or t - st.last_seen >= self.finalize_after_s:
                del self.state[tid]
                if st.frames >= MIN_FRAMES:
                    done.append(self._finish(tid, st))
        return done

    @staticmethod
    def _finish(tid: int, st: _State) -> Finished:
        cls = st.top_class()
        plate = st.top_plate()
        record = VehicleRecord(
            track_id=tid, vehicle_class=cls,
            class_conf=st.cls_sum[cls] / st.cls_n[cls] if st.cls_n else 0.0,
            plate_text=plate, plate_conf=st.plate_best[plate] if plate else 0.0,
            first_seen_s=st.first_seen, last_seen_s=st.last_seen, box=st.box)
        return Finished(record, st.crop, st.plate_box if plate else None)
