"""Layer 2: find things in a frame.

A detector is anything that takes an image and returns a list of Det. Swapping models
(or adding a second, independent vehicle model later) means creating another detector
object from its own weights and config; nothing else changes.
"""
from __future__ import annotations

import os
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .types import Det


class Detector(Protocol):
    names: Sequence[str]  # every class name the model can output

    def __call__(self, image: np.ndarray) -> List[Det]: ...


def results_to_dets(result, names: Sequence[str]) -> List[Det]:
    """Convert one Ultralytics result into our Det list (kept separate so it can be tested)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    return [Det(names[k], float(c), tuple(float(v) for v in b))
            for b, c, k in zip(xyxy, conf, cls)]


def check_classes(model_names: Sequence[str], wanted: Sequence[str]) -> List[str]:
    """Class names the config expects but the model does not have (empty list = all fine)."""
    return sorted(set(wanted) - set(model_names))


class YoloDetector:
    """Ultralytics YOLO model. Note: Ultralytics is AGPL-3.0, see the README licensing section."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 960,
                 device: Optional[str] = None):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        names = self.model.names
        self.names = list(names.values()) if isinstance(names, dict) else list(names)

    def __call__(self, image: np.ndarray) -> List[Det]:
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        return results_to_dets(result, self.names)