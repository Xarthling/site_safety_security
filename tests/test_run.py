import csv
import os

import cv2
import yaml

from evaluate import main as evaluate_main
from run import run_clip
from safety.storage import connect, list_events, start_run
from tests.helpers import SixPeopleDetector, make_video


def load_cfg(tmp_path):
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["paths"]["snapshots"] = str(tmp_path / "snaps")
    return cfg


def make_labels(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "camera_id", "event_type", "start_s", "end_s", "condition"])
        for kind, n in (("no_helmet", 3), ("no_vest", 2)):
            for _ in range(n):
                w.writerow(["gate.mp4", "gate_cam", kind, 0.0, 8.0, "day"])


def test_whole_pipeline_matches_the_labelled_scene(tmp_path, capsys):
    clips = tmp_path / "clips"
    clips.mkdir()
    video = str(clips / "gate.mp4")
    make_video(video, fps=10, seconds=8, size=(640, 480))
    cfg = load_cfg(tmp_path)
    db = str(tmp_path / "events.db")
    conn = connect(db)
    run_id = start_run(conn, "model", "config", ["gate.mp4"])

    det = SixPeopleDetector()
    stats = run_clip(video, det, cfg, camera_id="gate_cam", conn=conn, run_id=run_id,
                     annotated_path=str(tmp_path / "ann.mp4"))
    assert stats["frames"] == det.calls == 40
    assert stats["person_tracks"] == 6 and stats["short_tracks"] == 0
    assert stats["events"] == {"no_helmet": 3, "no_vest": 2}

    rows = list_events(conn, run_id=run_id)
    assert len(rows) == 5
    assert all(os.path.getsize(r["snapshot_path"]) > 0 for r in rows)
    assert os.path.exists(str(tmp_path / "ann.mp4"))
    conn.close()

    labels = str(tmp_path / "labels.csv")
    make_labels(labels)
    evaluate_main(["--labels", labels, "--db", db, "--clips-dir", str(clips)])
    out = capsys.readouterr().out
    total = [line for line in out.splitlines() if line.startswith("TOTAL")][0].split()
    # columns: group clips TP FN FP dup precision recall FA/hour delay_s
    assert total[2:6] == ["5", "0", "0", "0"]
    assert total[6:8] == ["1.00", "1.00"] and total[8] == "0.00"


def test_runs_without_a_database(tmp_path):
    video = str(tmp_path / "clip.mp4")
    make_video(video, fps=10, seconds=2, size=(640, 480))
    stats = run_clip(video, SixPeopleDetector(), load_cfg(tmp_path), max_frames=5)
    assert stats["frames"] == 5


# ---- gate camera: vehicle classification and licence plates --------------------------------

def gate_setup(tmp_path, **gate_override):
    from run import VehicleModels
    from safety.plates import PlateRead
    from tests.helpers import CarDetector, FakePlateReader
    video = str(tmp_path / "gate.mp4")
    make_video(video, fps=10, seconds=8, size=(640, 480))      # 40 processed frames at 5 fps
    cfg = load_cfg(tmp_path)
    cfg["gate"].update(gate_override)
    models = VehicleModels(CarDetector(first=2, last=20),
                           FakePlateReader([[PlateRead("ABC123", 0.9, (30.0, 20.0, 120.0, 60.0))]]))
    conn = connect(str(tmp_path / "e.db"))
    return video, cfg, models, conn, start_run(conn, "m", "c", ["gate.mp4"])


def test_gate_camera_logs_vehicle_class_and_plate(tmp_path):
    from safety.storage import list_vehicles
    video, cfg, models, conn, run_id = gate_setup(tmp_path)
    stats = run_clip(video, SixPeopleDetector(), cfg, camera_id="gate_cam", conn=conn, run_id=run_id,
                     vehicle_models=models, annotated_path=str(tmp_path / "ann.mp4"))
    assert stats["vehicles"] == [("car", "ABC123")]
    assert stats["events"] == {"no_helmet": 3, "no_vest": 2}          # PPE monitoring still works
    row = list_vehicles(conn, run_id=run_id)[0]
    assert row["camera_id"] == "gate_cam" and row["vehicle_class"] == "car" and row["plate"] == "ABC123"
    assert os.path.getsize(row["snapshot_path"]) > 0


def test_plate_can_be_stored_as_a_hash_and_is_blurred_in_the_snapshot(tmp_path):
    from safety.storage import list_vehicles
    video, cfg, models, conn, run_id = gate_setup(tmp_path, store_plate="hash")
    run_clip(video, SixPeopleDetector(), cfg, camera_id="gate_cam", conn=conn, run_id=run_id,
             vehicle_models=models)
    row = list_vehicles(conn, run_id=run_id)[0]
    assert row["plate"] != "ABC123" and len(row["plate"]) == 12
    assert os.path.getsize(row["snapshot_path"]) > 0


def test_plate_is_not_kept_when_store_plate_is_none(tmp_path):
    from safety.storage import list_vehicles
    video, cfg, models, conn, run_id = gate_setup(tmp_path, store_plate="none")
    run_clip(video, SixPeopleDetector(), cfg, camera_id="gate_cam", conn=conn, run_id=run_id,
             vehicle_models=models)
    assert list_vehicles(conn, run_id=run_id)[0]["plate"] is None


def test_other_cameras_do_not_run_the_vehicle_monitor(tmp_path):
    video, cfg, models, conn, run_id = gate_setup(tmp_path)
    stats = run_clip(video, SixPeopleDetector(), cfg, camera_id="yard_cam", conn=conn, run_id=run_id,
                     vehicle_models=models)
    assert stats["vehicles"] == [] and models.detector.n == 0   # the vehicle model was never called
