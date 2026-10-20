"""Run the pipeline (ingest -> perception -> tracking -> rules -> storage) on clips.

One invocation = one run: everything from all clips shares a run ID, so it can be scored together.
The camera named in config.yaml under `gate:` also gets vehicle classification and licence plates.

    python run.py --video data/clips/site_a_gate.mp4 --camera-id gate_cam --annotate
    python run.py --labels data/labels.csv --annotate       # every clip listed in labels.csv
    python evaluate.py                                      # then score the latest run

live.py reuses CameraPipeline, save_event, save_vehicle and the build_* helpers from this file.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import yaml

from safety import annotate
from safety.ingest import probe, read_frames
from safety.perception import Detector, YoloDetector, check_classes
from safety.plates import PlateReader, plate_for_storage
from safety.preprocess import PreprocessedDetector, PreprocessConfig
from safety.rules import FrameResult, Rules
from safety.storage import add_event, add_vehicle, connect, file_hash, start_run
from safety.tracking import Tracker
from safety.types import Track
from safety.vehicles import Finished, VehicleMonitor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VehicleModels(NamedTuple):
    """The shared models behind the gate camera's vehicle monitor (loaded once)."""
    detector: Detector
    reader: Optional[object]   # PlateReader, or None when read_plates is false


@dataclass
class FrameOutput:
    ppe: list                  # non-person detections from the PPE model
    tracks: List[Track]        # tracked people
    result: FrameResult        # PPE statuses and any events confirmed on this frame
    vehicles: List[Track] = field(default_factory=list)          # tracked vehicles (gate camera only)
    vehicle_labels: Dict[int, str] = field(default_factory=dict) # track_id -> e.g. "car ABC123"
    finished: List[Finished] = field(default_factory=list)       # vehicles logged on this frame


class CameraPipeline:
    """Everything that keeps state for ONE camera: its tracker, rules timers and, for the gate
    camera, its vehicle monitor. Use one per camera so cameras never mix."""

    def __init__(self, sample_fps: float, cfg: dict, vehicle_monitor: Optional[VehicleMonitor] = None):
        t = cfg["tracking"]
        self.person_class = cfg["detector"]["person_class"]
        self.tracker = Tracker("person", sample_fps, t["activation_threshold"], t["min_consecutive_frames"])
        self.rules = Rules.from_config(cfg["rules"])
        self.vehicle_monitor = vehicle_monitor

    def process(self, dets, video_time_s: float, image=None) -> FrameOutput:
        people = [d for d in dets if d.cls == self.person_class]
        ppe = [d for d in dets if d.cls != self.person_class]
        tracks = self.tracker.update(people)
        out = FrameOutput(ppe, tracks, self.rules.update(tracks, ppe, video_time_s))
        if self.vehicle_monitor is not None:
            v = self.vehicle_monitor.process(image, video_time_s)
            out.vehicles, out.vehicle_labels, out.finished = v.tracks, v.labels, v.finished
        return out

    def flush(self, video_time_s: float) -> List[Finished]:
        """End of clip: log the vehicles still in view."""
        return self.vehicle_monitor.flush(video_time_s) if self.vehicle_monitor else []


def default_clip_start(video: str) -> datetime:
    """Recorded clip: the file's modified time stands in for the recording start."""
    return datetime.fromtimestamp(os.path.getmtime(video), tz=timezone.utc)


def save_event(conn, run_id: int, ev, image, out: FrameOutput, camera_id: str,
               clip_name: str, clip_start: datetime, snapshot_dir: str) -> str:
    """Save the snapshot image and the database row for one PPE event."""
    os.makedirs(snapshot_dir, exist_ok=True)
    snap = annotate.draw(image, out.ppe, out.tracks, ev.event_type, out.result.statuses, ev.track_id)
    path = os.path.join(snapshot_dir, f"run{run_id}_{camera_id}_{ev.event_type}"
                                      f"_t{ev.track_id}_{ev.video_time_s:07.2f}.jpg")
    cv2.imwrite(path, snap)
    add_event(conn, run_id, ev, camera_id, clip_name, clip_start, path)
    return path


def save_vehicle(conn, run_id: int, fin: Finished, camera_id: str, clip_name: str, clip_start: datetime,
                 snapshot_dir: str, gate_cfg: dict) -> str:
    """Log one vehicle. The plate is stored as text, a hash or not at all (gate.store_plate); when it
    is not stored as text, the plate is also blurred in the saved picture."""
    mode, salt = gate_cfg.get("store_plate", "text"), gate_cfg.get("plate_salt", "")
    rec = fin.record
    path = ""
    if fin.crop is not None:
        crop = fin.crop
        if mode != "text" and fin.plate_box is not None:
            crop = annotate.blur_box(crop, fin.plate_box)
        os.makedirs(snapshot_dir, exist_ok=True)
        path = os.path.join(snapshot_dir, f"run{run_id}_{camera_id}_vehicle_t{rec.track_id}_{rec.first_seen_s:07.2f}.jpg")
        cv2.imwrite(path, crop)
    add_vehicle(conn, run_id, rec, camera_id, clip_name, clip_start,
                plate_for_storage(rec.plate_text, mode, salt), path)
    return path


def run_clip(video: str, detector: Detector, cfg: dict, *, camera_id: str = "cam1",
             clip_start: Optional[datetime] = None, conn=None, run_id: Optional[int] = None,
             annotated_path: Optional[str] = None, max_frames: Optional[int] = None,
             vehicle_models: Optional[VehicleModels] = None) -> dict:
    """Process one clip. If conn and run_id are given, events, vehicles and snapshots are saved.
    vehicle_models only takes effect when camera_id is the gate camera. Returns simple statistics."""
    target_fps = cfg["video"]["target_fps"]
    info = probe(video, target_fps)
    clip_name = os.path.basename(video)
    clip_start = clip_start or default_clip_start(video)
    gate = cfg.get("gate")
    monitor = None
    if vehicle_models is not None and gate and camera_id == gate["camera_id"]:
        monitor = VehicleMonitor(vehicle_models.detector, vehicle_models.reader, gate, info.sample_fps)
    camera = CameraPipeline(info.sample_fps, cfg, monitor)

    writer = None
    if annotated_path:
        os.makedirs(os.path.dirname(os.path.abspath(annotated_path)), exist_ok=True)
        writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.sample_fps, (info.width, info.height))

    frames_seen = Counter()   # track_id -> processed frames it appeared in
    events_by_type, vehicles_logged = Counter(), []
    n_frames, last_t = 0, 0.0

    def log_vehicles(finished: List[Finished]) -> None:
        for fin in finished:
            vehicles_logged.append(fin.record)
            if conn is not None:
                save_vehicle(conn, run_id, fin, camera_id, clip_name, clip_start, cfg["paths"]["snapshots"], gate)

    for frame in read_frames(video, target_fps, max_frames):
        out = camera.process(detector(frame.image), frame.video_time_s, frame.image)
        for t in out.tracks:
            frames_seen[t.track_id] += 1
        for ev in out.result.events:
            events_by_type[ev.event_type] += 1
            if conn is not None:
                save_event(conn, run_id, ev, frame.image, out, camera_id, clip_name, clip_start,
                           cfg["paths"]["snapshots"])
        log_vehicles(out.finished)
        if writer is not None:
            caption = f"t={frame.video_time_s:6.1f}s  people={len(out.tracks)}"
            writer.write(annotate.draw(frame.image, out.ppe, out.tracks, caption, out.result.statuses,
                                       vehicles=out.vehicles, vehicle_labels=out.vehicle_labels))
        n_frames, last_t = n_frames + 1, frame.video_time_s
    log_vehicles(camera.flush(last_t))
    if writer is not None:
        writer.release()

    return {"clip": clip_name, "frames": n_frames, "sample_fps": info.sample_fps,
            "person_tracks": len(frames_seen),
            "short_tracks": sum(1 for n in frames_seen.values() if n < 3),
            "events": dict(events_by_type),
            "vehicles": [(v.vehicle_class, v.plate_text) for v in vehicles_logged]}


def clips_from_labels(labels_path: str, clips_dir: str) -> List[Tuple[str, str]]:
    """(path, camera_id) for every clip listed in labels.csv."""
    from evaluate import read_labels
    seen = {}
    for row in read_labels(labels_path):
        seen.setdefault(row["clip"], row["camera_id"])
    return [(os.path.join(clips_dir, clip), cam or "cam1") for clip, cam in seen.items()]


def make_jobs(videos: List[str], labels: Optional[str], clips_dir: str, camera_id: str) -> List[Tuple[str, str]]:
    """(path, camera_id) pairs from --video and/or --labels."""
    jobs = [(v, camera_id) for v in videos]
    if labels:
        jobs += clips_from_labels(labels, clips_dir)
    return jobs


def build_detector(cfg: dict, device: Optional[str] = None) -> YoloDetector:
    """Load the PPE model and check that every class name in config.yaml exists in it."""
    d, r = cfg["detector"], cfg["rules"]
    detector = YoloDetector(d["weights"], d["conf"], d["imgsz"], device)
    print("PPE model classes:", detector.names)
    wanted = [d["person_class"]] + [c for k in ("helmet", "vest") if k in r
                                    for c in r[k]["present"] + r[k].get("absent", [])]
    missing = check_classes(detector.names, wanted)
    if missing:
        raise SystemExit(f"config.yaml expects classes the PPE model does not have: {missing}. "
                         f"Fix them to match the list above.")
    return PreprocessedDetector(detector, PreprocessConfig.from_config(cfg))


def build_vehicle_models(cfg: dict, jobs: List[Tuple[str, str]], device: Optional[str] = None) -> Optional[VehicleModels]:
    """Load the vehicle model and plate reader, but only if some clip belongs to the gate camera."""
    gate = cfg.get("gate")
    if not gate or not any(cam == gate["camera_id"] for _, cam in jobs):
        return None
    detector = YoloDetector(gate["vehicle_weights"], gate["conf"], gate["imgsz"], device)
    print("Vehicle model classes:", detector.names)
    missing = check_classes(detector.names, gate["vehicle_classes"])
    if missing:
        raise SystemExit(f"gate.vehicle_classes has names the vehicle model does not have: {missing}.")
    reader = PlateReader(gate["plate_detector"], gate["plate_ocr"]) if gate["read_plates"] else None
    print(f"Gate camera '{gate['camera_id']}': vehicles + "
          f"{'licence plates (stored as ' + gate['store_plate'] + ')' if reader else 'classification only'}")
    return VehicleModels(PreprocessedDetector(detector, PreprocessConfig.from_config(cfg)), reader)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="*", default=[], help="one or more clip files")
    ap.add_argument("--labels", help="run every clip listed in this labels.csv")
    ap.add_argument("--clips-dir", default="data/clips", help="where --labels clips live")
    ap.add_argument("--camera-id", default="cam1", help="camera for --video clips")
    ap.add_argument("--clip-start", help="real recording start, ISO 8601 with timezone")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--annotate", action="store_true", help="write annotated videos to outputs/annotated/")
    ap.add_argument("--max-frames", type=int, help="stop each clip after N processed frames")
    ap.add_argument("--device", help="e.g. cpu, 0, mps")
    ap.add_argument("--notes", default="", help="free text stored with the run")
    args = ap.parse_args()

    jobs = make_jobs(args.video, args.labels, args.clips_dir, args.camera_id)
    if not jobs:
        raise SystemExit("Nothing to do: pass --video <file> or --labels data/labels.csv")

    cfg = load_config(args.config)
    detector = build_detector(cfg, args.device)
    vehicle_models = build_vehicle_models(cfg, jobs, args.device)

    clip_start = datetime.fromisoformat(args.clip_start) if args.clip_start else None
    conn = connect(cfg["paths"]["db"])
    run_id = start_run(conn, file_hash(cfg["detector"]["weights"]), file_hash(args.config),
                       [os.path.basename(p) for p, _ in jobs], args.notes)
    print(f"Run {run_id}: {len(jobs)} clip(s)")

    for path, camera in jobs:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipped (its labels will count as misses)")
            continue
        out = os.path.join("outputs", "annotated", os.path.basename(path)) if args.annotate else None
        s = run_clip(path, detector, cfg, camera_id=camera, clip_start=clip_start, conn=conn, run_id=run_id,
                     annotated_path=out, max_frames=args.max_frames, vehicle_models=vehicle_models)
        print(f"  {s['clip']}: {s['frames']} frames, {s['person_tracks']} person tracks "
              f"({s['short_tracks']} short), events {s['events'] or 'none'}")
        if s["vehicles"]:
            print(f"    vehicles logged: {s['vehicles']}")
    conn.close()
    print(f"\nDone. Score it with:  python evaluate.py --run-id {run_id}")


if __name__ == "__main__":
    main()