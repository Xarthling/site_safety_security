from datetime import datetime, timezone

import pytest

from safety.storage import add_event, connect, latest_run_id, list_events, set_review, start_run
from safety.storage import get_camera_config, list_camera_configs, upsert_camera_config
from safety.types import Event
from safety.configuration import CameraConfig, CameraSource
from safety.types import NO_HELMET, NO_VEST

START = datetime(2026, 10, 19, 8, 30, tzinfo=timezone.utc)


def make_event(kind=NO_HELMET, t=12.5):
    return Event(kind, track_id=7, video_time_s=t, confidence=0.83, box=(10, 20, 110, 220))


def test_run_and_event_round_trip(tmp_path):
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "model-abc", "config-123", ["a.mp4"])
    add_event(conn, run_id, make_event(), "cam1", "a.mp4", START, "snap.jpg")
    rows = list_events(conn, run_id=run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["event_type"] == NO_HELMET and row["camera_id"] == "cam1" and row["clip"] == "a.mp4"
    assert row["timestamp_utc"].startswith("2026-10-19T08:30:12")  # clip start + 12.5 s
    assert row["review_status"] == "unreviewed"
    assert latest_run_id(conn) == run_id


def test_filters(tmp_path):
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "m", "c", ["a.mp4"])
    add_event(conn, run_id, make_event(NO_HELMET), "cam1", "a.mp4", START, "s1.jpg")
    add_event(conn, run_id, make_event(NO_VEST, 20), "cam2", "a.mp4", START, "s2.jpg")
    assert len(list_events(conn)) == 2
    assert len(list_events(conn, event_type=NO_VEST)) == 1
    assert len(list_events(conn, camera_id="cam1")) == 1
    assert list_events(conn, run_id=run_id + 1) == []


def test_review_status(tmp_path):
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "m", "c", [])
    event_id = add_event(conn, run_id, make_event(), "cam1", "a.mp4", START, "s.jpg")
    set_review(conn, event_id, "false_positive", "shadow")
    row = list_events(conn, review_status="false_positive")[0]
    assert row["review_notes"] == "shadow" and row["reviewed_at"] is not None
    with pytest.raises(ValueError):
        set_review(conn, event_id, "maybe")
    with pytest.raises(ValueError):
        set_review(conn, 999, "confirmed")


def test_naive_clip_start_is_rejected(tmp_path):
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "m", "c", [])
    with pytest.raises(ValueError):
        add_event(conn, run_id, make_event(), "cam1", "a.mp4", datetime(2026, 10, 19, 8, 30), "s.jpg")


def test_camera_config_round_trip_and_revision(tmp_path):
    conn = connect(str(tmp_path / "e.db"))
    camera = CameraConfig(
        id="gate_cam",
        name="Main gate",
        enabled=True,
        source=CameraSource("recorded", "clips/gate.mp4"),
        capabilities=frozenset({"ppe", "vehicles", "licence_plates"}),
        zones={"detection_zone": [[0.0, 0.0], [1.0, 1.0]]},
        rules={"confirm_seconds": 3.0},
        vehicle={"read_plates": True, "store_plate": "text"},
    )

    assert upsert_camera_config(conn, camera) == 1
    loaded = get_camera_config(conn, "gate_cam")
    assert loaded == camera
    assert upsert_camera_config(conn, camera) == 2
    assert len(list_camera_configs(conn)) == 1
