import numpy as np
import pytest

from run import load_config
from safety.plates import PlateRead
from safety.storage import add_vehicle, connect, list_vehicles, start_run
from safety.types import VehicleRecord
from safety.vehicles import MAX_ATTEMPTS, VehicleMonitor, best_class, iou
from tests.helpers import CarDetector, FakePlateReader
from datetime import datetime, timezone

IMG = np.zeros((480, 640, 3), dtype=np.uint8)
FPS = 5
BOX = (30.0, 20.0, 120.0, 60.0)


def gate_cfg(**override):
    cfg = dict(load_config("config.yaml")["gate"])
    cfg.update(override)
    return cfg


def read(text, conf):
    return [PlateRead(text, conf, BOX)]


def drive(monitor, frames):
    """Process `frames` frames; return (finished records, video time of the last frame)."""
    finished, t = [], 0.0
    for _ in range(frames):
        finished += monitor.process(IMG, t).finished
        t += 1.0 / FPS
    return finished, t


def test_iou_and_class_lookup():
    from safety.types import Det
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    dets = [Det("truck", .9, (0, 0, 100, 100)), Det("car", .8, (200, 200, 300, 300))]
    assert best_class((10, 10, 100, 100), dets) == "truck"
    assert best_class((500, 500, 600, 600), dets) == "vehicle"


def test_class_is_decided_by_vote_over_the_whole_pass():
    det = CarDetector(first=0, last=15, classes=("car", "car", "truck"))
    m = VehicleMonitor(det, None, gate_cfg(), FPS)
    done, _ = drive(m, 15 + 20)  # vehicle drives past, then 4 s of empty road
    assert len(done) == 1
    assert done[0].record.vehicle_class == "car" and done[0].record.plate_text is None
    assert done[0].record.first_seen_s < done[0].record.last_seen_s


def test_plate_is_decided_by_vote_and_reading_stops_when_settled():
    reader = FakePlateReader([read("A8C123", 0.7), read("ABC123", 0.9), read("ABC123", 0.85)])
    m = VehicleMonitor(CarDetector(first=0, last=15), reader, gate_cfg(), FPS)
    done, _ = drive(m, 15 + 20)
    rec = done[0].record
    assert rec.plate_text == "ABC123" and rec.plate_conf == pytest.approx(0.9)
    assert reader.calls == 3                      # stopped once ABC123 had been read twice
    assert done[0].crop is not None and done[0].plate_box == BOX


def test_small_vehicle_is_classified_but_no_plate_is_read():
    reader = FakePlateReader([read("ABC123", 0.9)])
    m = VehicleMonitor(CarDetector(first=0, last=15, width=100), reader, gate_cfg(min_vehicle_width_px=120), FPS)
    done, _ = drive(m, 35)
    assert reader.calls == 0 and done[0].record.plate_text is None and done[0].crop is not None


def test_low_confidence_reads_are_ignored_and_attempts_are_capped():
    reader = FakePlateReader([read("ABC123", 0.3)])
    m = VehicleMonitor(CarDetector(first=0, last=30), reader, gate_cfg(plate_min_conf=0.6), FPS)
    done, _ = drive(m, 30 + 20)
    assert done[0].record.plate_text is None
    assert reader.calls == MAX_ATTEMPTS


def test_plate_reading_can_be_switched_off():
    m = VehicleMonitor(CarDetector(first=0, last=15), None, gate_cfg(read_plates=False), FPS)
    assert drive(m, 35)[0][0].record.plate_text is None


def test_vehicle_is_logged_only_after_it_has_left_for_a_while():
    m = VehicleMonitor(CarDetector(first=0, last=10), None, gate_cfg(finalize_after_s=3.0), FPS)
    t, logged_at = 0.0, None
    for _ in range(60):
        if m.process(IMG, t).finished:
            logged_at = t
            break
        t += 1.0 / FPS
    assert logged_at is not None
    assert 3.0 <= logged_at - 9 / FPS <= 3.4      # about finalize_after_s after it was last seen


def test_ghost_tracks_are_not_logged():
    m = VehicleMonitor(CarDetector(first=0, last=2), None, gate_cfg(), FPS)
    assert drive(m, 30)[0] == []


def test_flush_logs_vehicles_still_in_view_at_the_end_of_the_clip():
    m = VehicleMonitor(CarDetector(first=0, last=999), None, gate_cfg(), FPS)
    done, t = drive(m, 12)
    assert done == [] and len(m.flush(t)) == 1 and m.flush(t) == []


def test_labels_follow_the_privacy_setting():
    reader = FakePlateReader([read("ABC123", 0.9)])
    m = VehicleMonitor(CarDetector(first=0, last=999), reader,
                       gate_cfg(store_plate="hash", plate_salt="s"), FPS)
    labels = {}
    for i in range(6):
        labels = m.process(IMG, i / FPS).labels or labels
    assert labels and all("ABC123" not in v and v.startswith("car ") for v in labels.values())
    m2 = VehicleMonitor(CarDetector(first=0, last=999), FakePlateReader([read("ABC123", 0.9)]),
                        gate_cfg(store_plate="text"), FPS)
    texts = [v for i in range(6) for v in m2.process(IMG, i / FPS).labels.values()]
    assert "car ABC123" in texts


def test_vehicle_log_storage(tmp_path):
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "m", "c", [])
    rec = VehicleRecord(3, "truck", 0.88, "ABC123", 0.91, 4.0, 9.5, (10, 20, 300, 260))
    start = datetime(2026, 10, 19, 8, 30, tzinfo=timezone.utc)
    add_vehicle(conn, run_id, rec, "gate_cam", "g.mp4", start, "ABC123", "snap.jpg")
    add_vehicle(conn, run_id, VehicleRecord(4, "car", 0.8, None, 0.0, 12.0, 15.0, (0, 0, 5, 5)),
                "gate_cam", "g.mp4", start, None)
    rows = list_vehicles(conn, run_id=run_id)
    assert [r["vehicle_class"] for r in rows] == ["truck", "car"]
    assert rows[0]["timestamp_utc"].startswith("2026-10-19T08:30:04") and rows[1]["plate"] is None
    assert len(list_vehicles(conn, plate="ABC123")) == 1 and len(list_vehicles(conn, vehicle_class="car")) == 1
    with pytest.raises(ValueError):
        add_vehicle(conn, run_id, rec, "gate_cam", "g.mp4", datetime(2026, 10, 19), "x")
