"""Licence plate reading, for the gate camera only.

PlateReader wraps fast-alpr: a small plate detector plus a text recogniser, both ONNX models that
download themselves on first use. Give it a picture (here: a crop around one vehicle) and it returns
the plates it can read.

Plates are personal information. plate_for_storage() applies the store_plate setting from config.yaml:
  text = keep the plate as read,  hash = keep only a salted fingerprint (can still spot repeat
  visits, cannot be read back),  none = do not keep plates at all.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .types import Box

MIN_LEN, MAX_LEN = 3, 9   # plates outside this length are treated as misreads


@dataclass(frozen=True)
class PlateRead:
    text: str    # normalised: upper-case letters and digits only
    conf: float
    box: Box     # position inside the image that was read


def normalise_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def ocr_confidence(conf, n_chars: int) -> float:
    """fast-alpr gives either one number or one number per character slot (padded); average the real ones."""
    if isinstance(conf, (list, tuple, np.ndarray)):
        values = [float(v) for v in list(conf)[:max(1, n_chars)]]
        return sum(values) / len(values) if values else 0.0
    return float(conf)


def plate_for_storage(text: Optional[str], mode: str, salt: str = "") -> Optional[str]:
    """What to keep for a plate: its text, a salted hash, or nothing."""
    if text is None or mode == "none":
        return None
    if mode == "text":
        return text
    if mode == "hash":
        return hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:12]
    raise ValueError(f"store_plate must be text, hash or none, got {mode!r}")


class PlateReader:
    def __init__(self, detector_model: str = "yolo-v9-t-384-license-plate-end2end",
                 ocr_model: str = "cct-xs-v2-global-model", ocr_device: str = "auto"):
        try:
            from fast_alpr import ALPR
        except ModuleNotFoundError as exc:
            if exc.name == "onnxruntime":
                raise ImportError(
                    "fast-alpr requires onnxruntime. Run: pip install onnxruntime"
                ) from exc
            if exc.name == "fast_alpr":
                raise ImportError(
                    "fast-alpr is not installed. Run: pip install fast-alpr"
                ) from exc
            raise
        self._alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model, ocr_device=ocr_device)

    def __call__(self, image: np.ndarray) -> List[PlateRead]:
        reads = []
        for r in self._alpr.predict(image):
            if r.ocr is None:
                continue
            text = normalise_plate(r.ocr.text)
            if not MIN_LEN <= len(text) <= MAX_LEN:
                continue
            b = r.detection.bounding_box
            reads.append(PlateRead(text, ocr_confidence(r.ocr.confidence, len(r.ocr.text)),
                                   (float(b.x1), float(b.y1), float(b.x2), float(b.y2))))
        return reads
