"""Compare ByteTrack and BoT-SORT on the same clip, using the same detections.

    python compare_trackers.py --video data/clips/c7g.mp4 --camera-id gate_cam

The model runs once per frame and both trackers get identical detections, so any difference is
down to the trackers. Run it on a clip where the camera tilts, pans or shakes (and on a still one).

How to read the table:
  tracks    how many different IDs the people received (fewer is better when the same people stay in view)
  short     IDs that lasted under 3 processed frames: flicker, or an ID that was immediately replaced
  longest   frames covered by the longest-lived ID
  churn     tracks divided by the most people seen at once. About 1.0 means every person kept one ID.
            People who really enter and leave push it above 1, so compare methods on the SAME clip.
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, Optional, Sequence

from run import build_detector, detector_config, load_config
from safety.ingest import probe, read_frames
from safety.tracking import make_tracker


def compare(video: str, detector, cfg: dict, methods: Sequence[str] = ("bytetrack", "botsort"),
            camera_id: str = "cam1", max_frames: Optional[int] = None) -> Dict[str, dict]:
    info = probe(video, cfg["video"]["target_fps"])
    person_class = detector_config(cfg)["person_class"]
    trackers = {}
    for m in methods:
        variant = {**cfg, "tracking": {**cfg["tracking"], "method": m}, "cameras": {}}   # force this method
        trackers[m] = make_tracker("person", info.sample_fps, variant, camera_id)
    seen = {m: Counter() for m in methods}
    most_at_once = {m: 0 for m in methods}
    frames = 0
    for frame in read_frames(video, cfg["video"]["target_fps"], max_frames):
        people = [d for d in detector(frame.image) if d.cls == person_class]   # model runs once
        for m, tracker in trackers.items():
            tracks = tracker.update(people, frame.image)
            seen[m].update(t.track_id for t in tracks)
            most_at_once[m] = max(most_at_once[m], len(tracks))
        frames += 1
    return {m: {"frames": frames, "tracks": len(seen[m]), "short": sum(1 for n in seen[m].values() if n < 3),
                "longest": max(seen[m].values(), default=0),
                "churn": round(len(seen[m]) / max(1, most_at_once[m]), 2)} for m in methods}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera-id", default="cam1")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--device")
    a = ap.parse_args()
    cfg = load_config(a.config)
    result = compare(a.video, build_detector(cfg, a.device), cfg, camera_id=a.camera_id, max_frames=a.max_frames)
    print(f"\n{'method':<10}{'frames':>8}{'tracks':>8}{'short':>7}{'longest':>9}{'churn':>7}")
    for method, r in result.items():
        print(f"{method:<10}{r['frames']:>8}{r['tracks']:>8}{r['short']:>7}{r['longest']:>9}{r['churn']:>7}")


if __name__ == "__main__":
    main()
