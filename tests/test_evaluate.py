import csv
from datetime import datetime, timezone

import pytest

from evaluate import main, read_labels, score_clip
from safety.storage import add_event, connect, start_run
from safety.types import NO_HELMET, Event


def label(start=10.0, end=30.0, kind="no_helmet", cam="c1"):
    return {"clip": "a.mp4", "camera_id": cam, "event_type": kind, "start_s": start, "end_s": end}


def event(t, kind="no_helmet", cam="c1"):
    return {"event_type": kind, "camera_id": cam, "video_time_s": t}


def test_matching_event_is_true_positive_with_delay():
    r = score_clip([label()], [event(13.0)], tolerance=2.0)
    assert (r["tp"], r["fn"], r["fp"]) == (1, 0, 0)
    assert r["delays"] == [3.0]


def test_missed_violation_is_false_negative():
    r = score_clip([label()], [], tolerance=2.0)
    assert (r["tp"], r["fn"], r["fp"]) == (0, 1, 0)


def test_event_outside_window_is_false_alarm():
    r = score_clip([label(10, 30)], [event(50.0)], tolerance=2.0)
    assert (r["tp"], r["fn"], r["fp"], r["dup"]) == (0, 1, 1, 0)


def test_tolerance_widens_the_window():
    assert score_clip([label(10, 30)], [event(32.0)], tolerance=2.0)["tp"] == 1
    assert score_clip([label(10, 30)], [event(32.1)], tolerance=2.0)["tp"] == 0


def test_second_event_for_same_violation_is_a_duplicate():
    r = score_clip([label()], [event(13.0), event(20.0)], tolerance=2.0)
    assert (r["tp"], r["fp"], r["dup"]) == (1, 1, 1)


def test_wrong_type_or_camera_does_not_match():
    assert score_clip([label()], [event(13.0, kind="no_vest")], 2.0)["tp"] == 0
    assert score_clip([label()], [event(13.0, cam="c2")], 2.0)["tp"] == 0


def test_read_labels_validates(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("clip,camera_id,event_type,start_s,end_s\na.mp4,c1,no_helmet,abc,5\n")
    with pytest.raises(ValueError):
        read_labels(str(path))
    path.write_text("clip,event_type\na.mp4,none\n")
    with pytest.raises(ValueError):
        read_labels(str(path))


def test_full_run_prints_a_report(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "camera_id", "event_type", "start_s", "end_s", "condition"])
        w.writerow(["a.mp4", "c1", "no_helmet", 10, 30, "day"])
        w.writerow(["b.mp4", "c1", "none", "", "", "night"])
    db = str(tmp_path / "e.db")
    conn = connect(db)
    run_id = start_run(conn, "m", "c", ["a.mp4", "b.mp4"])
    start = datetime(2026, 10, 19, tzinfo=timezone.utc)
    add_event(conn, run_id, Event(NO_HELMET, 1, 13.0, 0.9, (0, 0, 1, 1)), "c1", "a.mp4", start, "s.jpg")
    add_event(conn, run_id, Event(NO_HELMET, 2, 5.0, 0.9, (0, 0, 1, 1)), "c1", "b.mp4", start, "s.jpg")
    main(["--labels", str(labels), "--db", db, "--clips-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "TOTAL" in out and "day" in out and "night" in out
    assert "n/a" in out  # clip lengths unknown here, so false alarms per hour is n/a


def test_details_flag_lists_hits_misses_and_false_alarms(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "camera_id", "event_type", "start_s", "end_s", "condition"])
        w.writerow(["a.mp4", "c1", "no_helmet", 10, 30, "day"])
        w.writerow(["a.mp4", "c1", "no_vest", 10, 30, "day"])
    db = str(tmp_path / "e.db")
    conn = connect(db)
    run_id = start_run(conn, "m", "c", ["a.mp4"])
    start = datetime(2026, 10, 19, tzinfo=timezone.utc)
    add_event(conn, run_id, Event(NO_HELMET, 1, 13.0, 0.9, (0, 0, 1, 1)), "c1", "a.mp4", start, "s.jpg")
    add_event(conn, run_id, Event(NO_HELMET, 2, 50.0, 0.6, (0, 0, 1, 1)), "c1", "a.mp4", start, "s.jpg")
    main(["--labels", str(labels), "--db", db, "--clips-dir", str(tmp_path), "--details"])
    out = capsys.readouterr().out
    assert "HIT    no_helmet" in out and "MISS   no_vest" in out and "FALSE  no_helmet" in out
