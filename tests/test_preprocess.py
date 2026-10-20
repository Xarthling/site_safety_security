import cv2
import numpy as np

from safety.preprocess import PreprocessedDetector, PreprocessConfig, preprocess_frame
from safety.types import Det


def test_disabled_preprocessing_returns_original_frame():
    image = np.full((10, 20, 3), 100, dtype=np.uint8)
    out, scale = preprocess_frame(image, PreprocessConfig())
    assert out is image and scale == 1.0


def test_preprocessing_preserves_shape_without_upscale():
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    out, scale = preprocess_frame(
        image, PreprocessConfig(enabled=True, clahe_clip=1.5, sharpen_amount=0.2)
    )
    assert out.shape == image.shape and scale == 1.0


def test_detector_boxes_are_mapped_back_after_upscale():
    class FakeDetector:
        names = ["person"]

        def __call__(self, image):
            assert image.shape[:2] == (20, 30)
            return [Det("person", 0.9, (4.0, 6.0, 18.0, 16.0))]

    detector = PreprocessedDetector(
        FakeDetector(), PreprocessConfig(enabled=True, upscale=2.0)
    )
    reads = detector(np.zeros((10, 15, 3), dtype=np.uint8))
    assert reads[0].box == (2.0, 3.0, 9.0, 8.0)
