import yaml

from safety.rules import Item, Rules, assess, assign_dets
from safety.types import MISSING, NO_HELMET, NO_VEST, OK, UNKNOWN, Det, Track

HELMET = Item("helmet", NO_HELMET, ("helmet",), ("no-helmet",), (0.0, 0.4))
VEST = Item("vest", NO_VEST, ("vest",), ("no-vest",), (0.15, 0.75))
BOX = (100.0, 100.0, 180.0, 300.0)   # 200 px tall
HEAD = (120.0, 100.0, 160.0, 140.0)  # centre inside the top 40% of BOX
TORSO = (110.0, 150.0, 170.0, 230.0)


def person(tid=1, box=BOX, conf=0.9):
    return Track(tid, "person", conf, box)


def rules(items=(HELMET,), confirm=3.0, clear=2.0, min_h=80):
    return Rules(items, confirm, clear, min_h)


def run(r, seconds, dets_at, fps=5, people=None):
    """Feed the rules one frame per 1/fps seconds; return all events."""
    events = []
    for i in range(int(seconds * fps)):
        t = i / fps
        events += r.update(people or [person()], dets_at(t), t).events
    return events


# ---- status of one item -------------------------------------------------------------------

def test_helmet_seen_is_ok():
    assert assess(person(), [Det("helmet", .8, HEAD)], HELMET, 80)[0] == OK


def test_no_helmet_class_is_missing():
    assert assess(person(), [Det("no-helmet", .7, HEAD)], HELMET, 80)[0] == MISSING


def test_higher_confidence_wins_when_both_seen():
    both = [Det("helmet", .6, HEAD), Det("no-helmet", .9, HEAD)]
    assert assess(person(), both, HELMET, 80)[0] == MISSING


def test_nothing_seen_is_unknown_when_model_has_a_missing_class():
    assert assess(person(), [], HELMET, 80)[0] == UNKNOWN


def test_nothing_seen_is_missing_when_inferring():
    inferred = Item("helmet", NO_HELMET, ("helmet",), (), (0.0, 0.4))
    assert assess(person(), [], inferred, 80)[0] == MISSING


def test_small_person_is_unknown_never_a_violation():
    small = person(box=(100, 100, 130, 150))  # 50 px tall
    assert assess(small, [Det("no-helmet", .9, (105, 102, 125, 115))], HELMET, 80)[0] == UNKNOWN


# ---- assigning detections to people -------------------------------------------------------

def test_detection_outside_the_item_band_is_ignored():
    feet = Det("helmet", .9, (120, 260, 160, 290))
    assert assign_dets([person()], [feet], HELMET)[1] == []


def test_overlapping_neighbours_do_not_share_a_helmet():
    a = person(1, (100, 100, 180, 300))
    b = person(2, (150, 100, 230, 300))        # overlaps a between x=150 and x=180
    helmet = Det("helmet", .9, (140, 100, 180, 140))  # centre x=160, inside both boxes
    got = assign_dets([a, b], [helmet], HELMET)
    assert len(got[1]) == 1 and got[2] == []   # goes to the nearer person only


# ---- timing -------------------------------------------------------------------------------

def test_one_event_per_episode_not_per_frame():
    events = run(rules(), 10, lambda t: [Det("no-helmet", .8, HEAD)])
    assert len(events) == 1
    assert 2.9 <= events[0].video_time_s <= 3.3
    assert events[0].event_type == NO_HELMET and events[0].track_id == 1


def test_short_violation_does_not_fire():
    assert run(rules(), 2, lambda t: [Det("no-helmet", .8, HEAD)]) == []


def test_brief_flicker_does_not_reset_the_timer():
    def dets(t):
        return [Det("helmet", .8, HEAD)] if 1.0 <= t < 1.4 else [Det("no-helmet", .8, HEAD)]
    assert len(run(rules(), 6, dets)) == 1


def test_rearms_after_the_helmet_goes_on_and_off_again():
    def dets(t):  # missing 0-4 s, worn 4-8 s, missing again 8-12 s
        return [Det("helmet", .8, HEAD)] if 4 <= t < 8 else [Det("no-helmet", .8, HEAD)]
    events = run(rules(), 12, dets)
    assert len(events) == 2 and events[0].video_time_s < 4 <= 8 < events[1].video_time_s


def test_long_unknown_gap_restarts_the_timer():
    r = rules(confirm=3.0, clear=1.0)
    def dets(t):  # missing 0-2 s, nothing seen 2-5 s, missing 5-7 s
        return [Det("no-helmet", .8, HEAD)] if (t < 2 or 5 <= t < 7) else []
    assert run(r, 7, dets) == []


def test_state_is_forgotten_for_vanished_tracks():
    r = rules()
    r.update([person(7)], [Det("no-helmet", .8, HEAD)], 0.0)
    assert (7, "helmet") in r.timer.state
    r.update([], [], 100.0)
    assert (7, "helmet") not in r.timer.state


# ---- a whole scene, like the labelled test clip ------------------------------------------

def test_six_people_scene_gives_exactly_the_expected_events():
    """3 fully kitted, 1 vest but no helmet, 2 with neither -> 3 no_helmet + 2 no_vest."""
    people = [person(i, (50.0 + 90 * (i - 1), 100.0, 110.0 + 90 * (i - 1), 300.0)) for i in range(1, 7)]

    def dets_for(i):
        x = 50.0 + 90 * (i - 1)
        head, torso = (x + 10, 100.0, x + 50, 140.0), (x + 5, 150.0, x + 55, 230.0)
        if i <= 3:
            return [Det("helmet", .9, head), Det("vest", .9, torso)]       # fully kitted
        if i == 4:
            return [Det("no-helmet", .8, head), Det("vest", .9, torso)]    # vest, no helmet
        return [Det("no-helmet", .8, head), Det("no-vest", .8, torso)]     # neither

    all_dets = [d for i in range(1, 7) for d in dets_for(i)]
    events = run(rules((HELMET, VEST)), 8, lambda t: all_dets, people=people)
    helmet_ids = sorted(e.track_id for e in events if e.event_type == NO_HELMET)
    vest_ids = sorted(e.track_id for e in events if e.event_type == NO_VEST)
    assert helmet_ids == [4, 5, 6]
    assert vest_ids == [5, 6]


# ---- config -------------------------------------------------------------------------------

def test_rules_are_built_from_the_config_file():
    cfg = yaml.safe_load(open("config.yaml"))["rules"]
    r = Rules.from_config(cfg)
    assert [i.name for i in r.items] == ["helmet", "vest"]
    assert r.items[0].absent == ("no-helmet",) and r.items[1].region == (0.15, 0.75)
    assert r.timer.confirm_s == 3.0 and r.min_height == 80
    del cfg["vest"]
    assert [i.name for i in Rules.from_config(cfg).items] == ["helmet"]
