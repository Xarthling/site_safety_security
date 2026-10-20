"""Layer 3: give each detected object a stable ID across frames (ByteTrack).

Use one Tracker per kind of object per camera (for example one for people and, later,
one for vehicles) so their IDs never mix. IDs are only meaningful within one camera and
one clip. A brand-new object gets no ID until it has been seen in a couple of frames,
which filters out one-frame false detections.
"""
from __future__ import annotations

from typing import List

import numpy as np
import supervision as sv

from .types import Det, Track


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
                 min_consecutive_frames: int = 2):
        """label is stored on every Track ('person' or 'vehicle'); fps is the *processed*
        frame rate (VideoInfo.sample_fps), not the file's own rate."""
        self.label = label
        self._tracker = _make_bytetrack(fps, activation_threshold, min_consecutive_frames)

    def update(self, dets: List[Det]) -> List[Track]:
        """Feed one frame's detections (call once per processed frame, even if empty)."""
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