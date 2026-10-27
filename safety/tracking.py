"""Layer 3: give each detected object a stable ID across frames.

Two methods, chosen in config.yaml (tracking.method, or per camera with cameras.<id>.tracker):
  bytetrack  fast; assumes a FIXED camera. This is what runs by default.
  botsort    Ultralytics BoT-SORT. It also measures how the camera itself moved between frames
             (global motion compensation, GMC) and corrects every track for it, so a tilting, panning,
             swaying or shaking camera does not make people "jump" and swap IDs.

Use one Tracker per kind of object per camera (for example one for people and one for vehicles) so their
IDs never mix. IDs are only meaningful within one camera and one clip. A brand-new object gets no ID
until it has been seen in two frames, which filters out one-frame false detections.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import supervision as sv

from .types import Det, Track

METHODS = ("bytetrack", "botsort")

# Ultralytics' own names and defaults (cfg/trackers/botsort.yaml). Override any of them under tracking.botsort.
BOTSORT_DEFAULTS = dict(
    track_high_thresh=0.25,   # detections above this are matched first
    track_low_thresh=0.1,     # weaker detections (between this and the high threshold) are matched second
    new_track_thresh=0.25,    # a detection must score at least this to start a new track
    match_thresh=0.8,         # association strictness (higher = more permissive matching)
    fuse_score=True,          # mix detection confidence into the matching cost
    gmc_method="sparseOptFlow",   # camera motion: sparseOptFlow | orb | sift | ecc | none
    proximity_thresh=0.5,     # ReID only
    appearance_thresh=0.8,    # ReID only
    with_reid=False,          # appearance matching. Not supported here: it needs the detector's own features
    model="auto",
)


def tracker_method(cfg: dict, camera_id: str = "") -> str:
    """Which method this camera uses: its own `tracker` setting under cameras:, else tracking.method."""
    camera = ((cfg.get("cameras") or {}).get(camera_id) or {})
    method = camera.get("tracker", cfg["tracking"].get("method", "bytetrack"))
    if method not in METHODS:
        raise ValueError(f"tracker must be one of {METHODS}, got {method!r}")
    return method


def make_tracker(label: str, fps: float, cfg: dict, camera_id: str = "") -> "Tracker":
    """A Tracker for one kind of object on one camera, set up from config.yaml."""
    t = cfg["tracking"]
    return Tracker(label, fps, t["activation_threshold"], t["min_consecutive_frames"],
                   tracker_method(cfg, camera_id), t.get("botsort"), t.get("lost_seconds", 2.0))


class _Dets:
    """The minimal detection set Ultralytics trackers read: conf, cls, xywh, xyxy, len, and selection by mask."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_Dets":
        return _Dets(self.xyxy[mask], self.conf[mask], self.cls[mask])


def botsort_buffer(lost_seconds: float, fps: float, takes_frame_rate: bool) -> int:
    """track_buffer for Ultralytics. Newer versions count it in processed frames; older versions
    count it in frames at 30 fps and rescale by the frame_rate argument."""
    return round(lost_seconds * (30 if takes_frame_rate else fps))


def _make_botsort(options: dict, fps: float, lost_seconds: float):
    try:
        from ultralytics.trackers.bot_sort import BOTSORT
    except ImportError:
        raise ImportError("tracker 'botsort' needs ultralytics. Run: pip install -U ultralytics")
    takes_rate = "frame_rate" in inspect.signature(BOTSORT.__init__).parameters
    opts = {**BOTSORT_DEFAULTS, **(options or {})}
    opts["track_buffer"] = botsort_buffer(lost_seconds, fps, takes_rate)
    args = SimpleNamespace(**opts)
    return BOTSORT(args, frame_rate=max(1, round(fps))) if takes_rate else BOTSORT(args)


def _make_bytetrack(fps: float, activation_threshold: float, min_consecutive_frames: int):
    """ByteTrack lives in the `trackers` package from supervision 0.27 onwards;
    older supervision versions bundle their own copy."""
    frame_rate = max(1.0, float(fps))
    try:
        from trackers import ByteTrackTracker
        return ByteTrackTracker(frame_rate=frame_rate,
                                track_activation_threshold=activation_threshold,
                                minimum_consecutive_frames=min_consecutive_frames)
    except ImportError:
        return sv.ByteTrack(frame_rate=int(round(frame_rate)))


class Tracker:
    def __init__(self, label: str, fps: float, activation_threshold: float = 0.4,
                 min_consecutive_frames: int = 2, method: str = "bytetrack",
                 botsort_options: Optional[dict] = None, lost_seconds: float = 2.0):
        """label is stored on every Track ('person' or 'vehicle'); fps is the *processed* frame rate
        (VideoInfo.sample_fps), not the file's own rate. activation_threshold and min_consecutive_frames
        apply to bytetrack; botsort uses botsort_options and lost_seconds (how long a lost track is kept
        alive, which sets how much occlusion it can ride out)."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.label, self.method = label, method
        if method == "botsort":
            self._tracker = _make_botsort(botsort_options or {}, fps, lost_seconds)
        else:
            self._tracker = _make_bytetrack(fps, activation_threshold, min_consecutive_frames)

    def update(self, dets: List[Det], image: Optional[np.ndarray] = None) -> List[Track]:
        """Feed one frame's detections (call once per processed frame, even if empty).
        botsort also needs the frame image, to measure camera motion."""
        if self.method == "botsort":
            return self._update_botsort(dets, image)
        if dets:
            sv_dets = sv.Detections(
                xyxy=np.array([d.box for d in dets], dtype=np.float32),
                confidence=np.array([d.conf for d in dets], dtype=np.float32),
                class_id=np.zeros(len(dets), dtype=int),
            )
        else:
            sv_dets = sv.Detections.empty()

        if hasattr(self._tracker, "update_with_detections"):  # older supervision.ByteTrack
            out = self._tracker.update_with_detections(sv_dets)
        else:
            out = self._tracker.update(sv_dets)

        tracks = []
        for i in range(len(out)):
            tid = out.tracker_id[i] if out.tracker_id is not None else None
            if tid is None or int(tid) < 0:  # -1 means "not yet confirmed as a track"
                continue
            conf = float(out.confidence[i]) if out.confidence is not None else 0.0
            tracks.append(Track(int(tid), self.label, conf,
                                tuple(float(v) for v in out.xyxy[i])))
        return tracks

    def _update_botsort(self, dets: List[Det], image: Optional[np.ndarray]) -> List[Track]:
        if image is None:
            raise ValueError("botsort needs the frame image to measure camera motion: call update(dets, image)")
        xyxy = np.array([d.box for d in dets], dtype=np.float32).reshape(-1, 4)
        conf = np.array([d.conf for d in dets], dtype=np.float32)
        rows = self._tracker.update(_Dets(xyxy, conf, np.zeros(len(dets), dtype=np.float32)), image)
        # rows are [x1, y1, x2, y2, track_id, score, cls, idx]; Ultralytics returns an empty 1-D array when nothing is tracked
        return [Track(int(r[4]), self.label, float(r[5]), (float(r[0]), float(r[1]), float(r[2]), float(r[3])))
                for r in (rows if len(rows) else [])]