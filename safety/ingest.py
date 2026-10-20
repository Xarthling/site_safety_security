"""Layer 1: turn a recorded clip into frames stamped with video time.

Frames are sampled at a lower rate than the camera records (target_fps), because the
rules work over seconds and every processed frame costs model time. Time is always
"seconds from the start of the clip"; real clock time is added later from the clip's
recording start (see storage.add_event).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

DEFAULT_FPS = 25.0  # used only if the file does not report a frame rate


@dataclass(frozen=True)
class VideoInfo:
    fps: float          # frame rate of the file
    width: int
    height: int
    frame_count: int    # 0 if the file does not report it
    duration_s: float   # 0.0 if frame_count is unknown
    stride: int         # process every Nth frame
    sample_fps: float   # frames per second actually processed


@dataclass(frozen=True)
class Frame:
    image: np.ndarray   # BGR image, as OpenCV reads it
    video_time_s: float # seconds from the start of the clip
    index: int          # frame number in the original file


def _open(path: str) -> cv2.VideoCapture:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV could not open {path}. Is it a valid video file?")
    return cap


def probe(path: str, target_fps: float = 5.0) -> VideoInfo:
    """Read the clip's properties and work out how often frames will be sampled."""
    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps or math.isnan(fps) or fps <= 0:
        fps = DEFAULT_FPS
    frame_count = max(frame_count, 0)
    stride = max(1, round(fps / target_fps))
    return VideoInfo(fps, width, height, frame_count,
                     frame_count / fps if frame_count else 0.0, stride, fps / stride)


def read_frames(path: str, target_fps: float = 5.0,
                max_frames: Optional[int] = None) -> Iterator[Frame]:
    """Yield sampled frames in order. max_frames limits how many are yielded (quick tests)."""
    info = probe(path, target_fps)
    cap = _open(path)
    index, yielded = 0, 0
    try:
        while max_frames is None or yielded < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            if index % info.stride == 0:
                yield Frame(image, index / info.fps, index)
                yielded += 1
            index += 1
    finally:
        cap.release()