"""Conservative image preprocessing before detector inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from .types import Det


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = False
    denoise_kernel: int = 0
    clahe_clip: float = 0.0
    clahe_grid: int = 8
    sharpen_amount: float = 0.0
    upscale: float = 1.0

    @classmethod
    def from_config(cls, config: dict) -> "PreprocessConfig":
        values = config.get("preprocess", {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            denoise_kernel=int(values.get("denoise_kernel", 0)),
            clahe_clip=float(values.get("clahe_clip", 0.0)),
            clahe_grid=int(values.get("clahe_grid", 8)),
            sharpen_amount=float(values.get("sharpen_amount", 0.0)),
            upscale=float(values.get("upscale", 1.0)),
        )


def preprocess_frame(image: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, float]:
    """Apply mild enhancement and return (image, scale used for its dimensions)."""
    if not config.enabled:
        return image, 1.0
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preprocess_frame expects a BGR color image")
    if config.denoise_kernel < 0 or config.clahe_clip < 0 or config.sharpen_amount < 0:
        raise ValueError("preprocessing strengths cannot be negative")
    if config.upscale < 1.0:
        raise ValueError("preprocess.upscale must be at least 1.0")

    out = image
    if config.denoise_kernel:
        kernel = config.denoise_kernel if config.denoise_kernel % 2 else config.denoise_kernel + 1
        out = cv2.GaussianBlur(out, (kernel, kernel), 0)
    if config.clahe_clip:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(config.clahe_clip, (max(1, config.clahe_grid),) * 2)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if config.sharpen_amount:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.0 + config.sharpen_amount, blurred,
                              -config.sharpen_amount, 0)
    scale = config.upscale
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return out, scale


class PreprocessedDetector:
    """Detector adapter that enhances inputs and maps boxes back to source coordinates."""

    def __init__(self, detector, config: PreprocessConfig):
        self.detector = detector
        self.config = config
        self.names: Sequence[str] = detector.names

    def __call__(self, image: np.ndarray) -> List[Det]:
        enhanced, scale = preprocess_frame(image, self.config)
        detections = self.detector(enhanced)
        if scale == 1.0:
            return detections
        return [
            Det(d.cls, d.conf, tuple(value / scale for value in d.box))
            for d in detections
        ]
