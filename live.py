"""Watch clips as a live CCTV wall: every clip plays at real speed as one camera in a tiled window,
with boxes, PPE status and an alert banner when a violation is confirmed.

    python live.py --labels data/labels.csv            # every clip in labels.csv, one tile each
    python live.py --video data/clips/a.mp4 data/clips/b.mp4
    python live.py --labels data/labels.csv --speed 2  # play twice as fast (a quick load test)

Keys: q = quit, space = pause / resume.

Like a real camera, a clip does not wait for the model. If processing cannot keep up, frames are
dropped and the tile shows LAG. Events are saved to the database exactly as in run.py
(use --no-save to just watch).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from run import (CameraPipeline, VehicleModels, build_detector, build_vehicle_models,
                 default_clip_start, load_config, make_jobs, save_event, save_vehicle)
from safety import annotate
from safety.ingest import probe
from safety.plates import plate_for_storage
from safety.storage import connect, file_hash, start_run
from safety.types import VEHICLE_IN, VEHICLE_OUT
from safety.vehicles import Finished, VehicleMonitor
from safety.zones import zones_for

RED, ORANGE = (40, 40, 220), (40, 140, 240)   # banner colours: violation, vehicle logged
BANNER_SECONDS = 4.0   # how long an alert stays on a tile
LAG_WARNING_S = 1.0    # show LAG when processing is this far behind the clip


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale an image to fit inside width x height, keeping its shape, on a black canvas."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def make_grid(tiles: List[np.ndarray], cols: int) -> np.ndarray:
    """Arrange same-sized tiles in rows of `cols`, padding the last row with black."""
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    tiles = tiles + [blank] * (rows * cols - len(tiles))
    return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])


class LiveCamera:
    """One clip playing as one camera."""

    def __init__(self, path: str, camera_id: str, cfg: dict, clip_start: datetime,
                 vehicle_models: Optional[VehicleModels] = None):
        self.camera_id, self.clip_name, self.clip_start = camera_id, os.path.basename(path), clip_start
        self.info = probe(path, cfg["video"]["target_fps"])
        self.cap = cv2.VideoCapture(path)
        gate = cfg.get("gate")
        self.zones = zones_for(cfg, camera_id, self.info.width, self.info.height)
        monitor = None
        if vehicle_models is not None and gate and camera_id == gate["camera_id"]:
            monitor = VehicleMonitor(vehicle_models.detector, vehicle_models.reader, gate,
                                     self.info.sample_fps, self.zones)
        self.pipeline = CameraPipeline(self.info.sample_fps, cfg, monitor, self.zones)
        self.vehicles, self.vehicle_labels = [], {}   # gate camera only
        self.index = 0                 # next frame number to read from the file
        self.image = None              # latest frame shown
        self.video_t = 0.0             # video time of that frame
        self.last_proc_t = -1e9        # video time of the last frame that went through the model
        self.ppe, self.tracks, self.statuses = [], [], {}
        self.banner, self.banner_until, self.banner_colour = "", -1.0, RED
        self.done = False
        self.n_processed = self.n_events = self.n_vehicles = 0


class LiveWall:
    """All cameras plus the shared detector. step(now) advances every camera to the playback
    clock `now` (seconds since start) and returns the tiled image, so it can be tested without
    a window or real waiting."""

    def __init__(self, jobs: List[Tuple[str, str]], detector, cfg: dict, conn=None,
                 run_id: Optional[int] = None, clip_start: Optional[datetime] = None,
                 tile_size: Tuple[int, int] = (640, 360), vehicle_models: Optional[VehicleModels] = None):
        self.detector, self.cfg, self.conn, self.run_id = detector, cfg, conn, run_id
        self.tile_w, self.tile_h = tile_size
        self.cameras: List[LiveCamera] = []
        for path, camera_id in jobs:
            if not os.path.exists(path):
                print(f"WARNING: {path} not found, skipped")
                continue
            self.cameras.append(LiveCamera(path, camera_id, cfg, clip_start or default_clip_start(path),
                                           vehicle_models))
        if not self.cameras:
            raise SystemExit("No clips could be opened.")
        self.cols = math.ceil(math.sqrt(len(self.cameras)))

    @property
    def all_done(self) -> bool:
        return all(c.done for c in self.cameras)

    def step(self, now: float) -> np.ndarray:
        for cam in self.cameras:
            self._advance(cam, now)
        return make_grid([self._tile(cam, now) for cam in self.cameras], self.cols)

    def _advance(self, cam: LiveCamera, now: float) -> None:
        if cam.done:
            return
        target = int(now * cam.info.fps)       # the frame that should be on screen right now
        if cam.image is not None and cam.index > target:
            return                              # already showing it
        while cam.index < target:               # behind: skip frames (dropped, never processed)
            if not cam.cap.grab():
                self._finish(cam, now)
                return
            cam.index += 1
        ok, image = cam.cap.read()
        if not ok:
            self._finish(cam, now)
            return
        cam.image, cam.video_t = image, cam.index / cam.info.fps
        cam.index += 1
        if cam.video_t - cam.last_proc_t >= 1.0 / cam.info.sample_fps - 1e-6:
            self._process(cam, now)

    def _finish(self, cam: LiveCamera, now: float) -> None:
        """The clip ended: log any vehicles still in view."""
        cam.done = True
        self._log_vehicles(cam, cam.pipeline.flush(cam.video_t), now)

    def _process(self, cam: LiveCamera, now: float) -> None:
        out = cam.pipeline.process(self.detector(cam.image), cam.video_t, cam.image)
        cam.ppe, cam.tracks, cam.statuses = out.ppe, out.tracks, out.result.statuses
        cam.vehicles, cam.vehicle_labels = out.vehicles, out.vehicle_labels
        cam.last_proc_t = cam.video_t
        cam.n_processed += 1
        for ev in out.all_events:
            cam.n_events += 1
            is_vehicle = ev.event_type in (VEHICLE_IN, VEHICLE_OUT)
            cam.banner = f"{ev.event_type.upper().replace('_', ' ')}  " + (ev.detail if is_vehicle else f"#{ev.track_id}")
            cam.banner_until, cam.banner_colour = now + BANNER_SECONDS, ORANGE if is_vehicle else RED
            if self.conn is not None:
                save_event(self.conn, self.run_id, ev, cam.image, out, cam.camera_id, cam.clip_name,
                           cam.clip_start, self.cfg["paths"]["snapshots"])
        self._log_vehicles(cam, out.finished, now)

    def _log_vehicles(self, cam: LiveCamera, finished: List[Finished], now: float) -> None:
        gate = self.cfg.get("gate", {})
        for fin in finished:
            cam.n_vehicles += 1
            plate = plate_for_storage(fin.record.plate_text, gate.get("store_plate", "text"), gate.get("plate_salt", ""))
            cam.banner = f"VEHICLE  {fin.record.vehicle_class}  {plate or 'no plate'}"
            cam.banner_until, cam.banner_colour = now + BANNER_SECONDS, ORANGE
            if self.conn is not None:
                save_vehicle(self.conn, self.run_id, fin, cam.camera_id, cam.clip_name, cam.clip_start,
                             self.cfg["paths"]["snapshots"], gate)

    def _tile(self, cam: LiveCamera, now: float) -> np.ndarray:
        w, h = self.tile_w, self.tile_h
        if cam.image is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = fit(annotate.draw(cam.image, cam.ppe, cam.tracks, "", cam.statuses,
                                       vehicles=cam.vehicles, vehicle_labels=cam.vehicle_labels,
                                       zones=cam.zones), w, h)
        cv2.rectangle(canvas, (0, 0), (w, 26), (30, 30, 30), -1)
        header = f"{cam.camera_id}   {cam.video_t:6.1f}s   people {len(cam.tracks)}"
        if cam.pipeline.vehicle_monitor is not None:
            header += f"   vehicles {len(cam.vehicles)}"
        cv2.putText(canvas, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        lag = now - cam.last_proc_t
        if cam.n_processed and not cam.done and lag > LAG_WARNING_S:
            cv2.putText(canvas, f"LAG {lag:.1f}s", (w - 120, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 140, 255), 2)
        if now < cam.banner_until:
            cv2.rectangle(canvas, (0, h - 40), (w, h), cam.banner_colour, -1)
            cv2.putText(canvas, cam.banner, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if cam.done:
            cv2.putText(canvas, "END OF CLIP", (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
        return canvas

    def summary(self, elapsed: float) -> List[str]:
        lines = []
        for c in self.cameras:
            rate = c.n_processed / elapsed if elapsed > 0 else 0.0
            note = "  <- below the target rate, the model is not keeping up" \
                if rate < 0.9 * c.info.sample_fps and not c.done else ""
            vehicles = f", {c.n_vehicles} vehicles" if c.pipeline.vehicle_monitor is not None else ""
            lines.append(f"  {c.camera_id}: {c.n_processed} frames processed ({rate:.1f}/s, "
                         f"target {c.info.sample_fps:.1f}/s), {c.n_events} events{vehicles}{note}")
        return lines

    def close(self) -> None:
        for c in self.cameras:
            c.cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="*", default=[], help="one or more clip files")
    ap.add_argument("--labels", help="use every clip listed in this labels.csv")
    ap.add_argument("--clips-dir", default="data/clips")
    ap.add_argument("--camera-id", default="cam1", help="camera name for --video clips")
    ap.add_argument("--clip-start", help="one shared recording start, ISO 8601 with timezone")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", help="e.g. cpu, 0, mps")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1.0 = real time")
    ap.add_argument("--tile-width", type=int, default=640)
    ap.add_argument("--tile-height", type=int, default=360)
    ap.add_argument("--no-save", action="store_true", help="watch only; do not write events")
    ap.add_argument("--notes", default="live wall")
    args = ap.parse_args()

    jobs = make_jobs(args.video, args.labels, args.clips_dir, args.camera_id)
    if not jobs:
        raise SystemExit("Nothing to watch: pass --video <files> or --labels data/labels.csv")
    cfg = load_config(args.config)
    detector = build_detector(cfg, args.device)
    vehicle_models = build_vehicle_models(cfg, jobs, args.device)

    conn = run_id = None
    if not args.no_save:
        conn = connect(cfg["paths"]["db"])
        run_id = start_run(conn, file_hash(cfg["detector"]["weights"]), file_hash(args.config),
                           [os.path.basename(p) for p, _ in jobs], args.notes)
        print(f"Run {run_id}: events are being saved")
    clip_start = datetime.fromisoformat(args.clip_start) if args.clip_start else None
    wall = LiveWall(jobs, detector, cfg, conn, run_id, clip_start, (args.tile_width, args.tile_height),
                    vehicle_models)

    title = "CCTV wall (q = quit, space = pause)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    clock, last, paused, finished_msg = 0.0, time.monotonic(), False, False
    try:
        while True:
            tick = time.monotonic()
            if not paused:
                clock += (tick - last) * args.speed
            last = tick
            cv2.imshow(title, wall.step(clock))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
            if wall.all_done and not finished_msg:
                print("All clips finished. Press q to close the window.")
                finished_msg = True
    finally:
        cv2.destroyAllWindows()
        print(f"\nPlayed {clock:.0f}s of footage:")
        print("\n".join(wall.summary(clock)))
        wall.close()
        if conn is not None:
            conn.close()
            print(f"\nScore it with:  python evaluate.py --run-id {run_id}")


if __name__ == "__main__":
    main()
