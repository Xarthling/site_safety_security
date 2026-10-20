from safety.tracking import Tracker
from safety.types import Det


def person(x, y=50, conf=0.9):
    return Det("person", conf, (x, y, x + 40, y + 120))


def run(tracker, frames):
    return [tracker.update(dets) for dets in frames]


def test_one_moving_person_keeps_one_id():
    t = Tracker("person", fps=5)
    out = run(t, [[person(20 + 4 * i)] for i in range(20)])
    ids = {tr.track_id for frame in out for tr in frame}
    assert len(ids) == 1
    assert all(tr.label == "person" for frame in out for tr in frame)
    assert len(out[-1]) == 1


def test_two_separated_people_get_different_ids():
    t = Tracker("person", fps=5)
    out = run(t, [[person(20 + 3 * i), person(200 - 3 * i)] for i in range(20)])
    ids = {tr.track_id for frame in out for tr in frame}
    assert len(ids) == 2
    assert len(out[-1]) == 2


def test_empty_frames_are_handled():
    t = Tracker("person", fps=5)
    assert t.update([]) == []
    out = run(t, [[person(30 + i)] for i in range(5)] + [[], []])
    assert out[-1] == []


def test_new_object_gets_no_id_on_its_first_frame():
    t = Tracker("person", fps=5)
    assert t.update([person(50)]) == []  # needs min_consecutive_frames before it is confirmed


def test_vehicle_label_is_carried_through():
    t = Tracker("vehicle", fps=5)
    out = run(t, [[Det("vehicle", 0.9, (10 + 5 * i, 10, 110 + 5 * i, 80))] for i in range(8)])
    assert any(tr.label == "vehicle" for frame in out for tr in frame)
