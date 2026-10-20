"""Small helpers shared by the tests."""
import cv2
import numpy as np

from safety.types import Det


def make_video(path, fps=10, seconds=3, size=(320, 240)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(fps * seconds):
        w.write(np.full((size[1], size[0], 3), 90, dtype=np.uint8))
    w.release()


class SixPeopleDetector:
    """The labelled scene: 3 fully kitted, 1 vest but no helmet, 2 with neither."""
    names = ["person", "helmet", "no-helmet", "vest", "no-vest"]

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        dets = []
        for i in range(6):
            x = 20.0 + 100 * i
            dets.append(Det("person", 0.9, (x, 100.0, x + 60, 300.0)))
            head, torso = (x + 10, 102.0, x + 50, 140.0), (x + 5, 150.0, x + 55, 230.0)
            dets.append(Det("helmet" if i < 3 else "no-helmet", 0.85, head))
            dets.append(Det("vest" if i < 4 else "no-vest", 0.85, torso))
        return dets


class CarDetector:
    """Scripted vehicle model: one vehicle drives left to right while first <= call number < last.
    classes cycles per call, e.g. ("car", "car", "truck")."""
    names = ["car", "truck", "bus", "motorcycle", "person"]

    def __init__(self, first=0, last=10**6, classes=("car",), width=200):
        self.n, self.first, self.last, self.classes, self.width = 0, first, last, classes, width

    def __call__(self, image):
        n, self.n = self.n, self.n + 1
        if not self.first <= n < self.last:
            return []
        x = 20.0 + 6 * n
        return [Det(self.classes[n % len(self.classes)], 0.9, (x, 120.0, x + self.width, 260.0))]


class FakePlateReader:
    """Scripted plate reader: returns script[0] on the first call, script[1] on the second, ..."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def __call__(self, crop):
        reads = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return reads
