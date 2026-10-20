"""Layer 4: turn tracked people and PPE detections into confirmed violation events.

Per frame, for each tracked person and each PPE item (helmet, vest):
  1. assign nearby PPE detections to the person (assign_dets)
  2. decide the status: ok, missing or unknown (assess)
  3. feed the status to a timer that confirms a violation once it has lasted long enough,
     and reports it only once per episode (ViolationTimer)

Times are video seconds, so recorded clips behave the same on every run.
This file has no model or video code: it works on plain Det / Track / Event objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .types import MISSING, NO_HELMET, NO_VEST, OK, UNKNOWN, Det, Event, Track

EVENT_TYPES = {"helmet": NO_HELMET, "vest": NO_VEST}


@dataclass(frozen=True)
class Item:
    """One PPE item to check."""
    name: str                          # "helmet" or "vest"
    event_type: str                    # event written when it is missing
    present: Tuple[str, ...]           # model classes meaning "item worn", e.g. ("helmet",)
    absent: Tuple[str, ...] = ()       # model classes meaning "item missing", e.g. ("no-helmet",)
    region: Tuple[float, float] = (0.0, 1.0)  # band of the person box (fractions from the top)


@dataclass
class FrameResult:
    events: List[Event]                                    # violations confirmed on this frame
    statuses: Dict[int, Dict[str, Tuple[str, float]]]      # track_id -> item -> (status, confidence)


def _center(box) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def assign_dets(persons: Sequence[Track], dets: Sequence[Det], item: Item) -> Dict[int, List[Det]]:
    """Give each PPE detection to ONE person: among the people whose expected body band contains
    the detection's centre, the one whose box is horizontally closest. This keeps a helmet from
    counting for two neighbours when their boxes overlap."""
    wanted = set(item.present) | set(item.absent)
    assigned: Dict[int, List[Det]] = {p.track_id: [] for p in persons}
    for d in dets:
        if d.cls not in wanted:
            continue
        cx, cy = _center(d.box)
        best = None  # (horizontal distance, track_id)
        for p in persons:
            x1, y1, x2, y2 = p.box
            h = y2 - y1
            in_band = y1 + item.region[0] * h <= cy <= y1 + item.region[1] * h
            if x1 <= cx <= x2 and in_band:
                dist = abs(cx - (x1 + x2) / 2.0)
                if best is None or dist < best[0]:
                    best = (dist, p.track_id)
        if best is not None:
            assigned[best[1]].append(d)
    return assigned


def assess(person: Track, assigned: Sequence[Det], item: Item,
           min_height: float) -> Tuple[str, float]:
    """Status of one item on one person: (ok | missing | unknown, confidence)."""
    if person.box[3] - person.box[1] < min_height:
        return UNKNOWN, person.conf  # too small to judge: never a violation

    def best(classes) -> float:
        confs = [d.conf for d in assigned if d.cls in classes]
        return max(confs) if confs else 0.0

    present, absent = best(item.present), best(item.absent)
    if present > 0 and present >= absent:
        return OK, present
    if absent > 0:
        return MISSING, absent
    if not item.absent:
        return MISSING, person.conf  # no "missing" class: infer from not seeing the item
    return UNKNOWN, person.conf      # the model saw nothing either way


@dataclass
class _State:
    bad_since: Optional[float] = None
    last_bad: Optional[float] = None
    last_seen: float = 0.0
    reported: bool = False


class ViolationTimer:
    """Confirms a violation after confirm_s of 'missing', once per episode.
    An episode ends after clear_s seconds with no 'missing', so brief detection flicker
    neither delays an event nor creates a second one."""

    def __init__(self, confirm_s: float, clear_s: float, prune_s: float = 30.0):
        self.confirm_s, self.clear_s, self.prune_s = confirm_s, clear_s, prune_s
        self.state: Dict[Tuple[int, str], _State] = {}

    def update(self, key: Tuple[int, str], status: str, t: float) -> bool:
        """Returns True on the frame where the violation becomes confirmed."""
        st = self.state.setdefault(key, _State())
        st.last_seen = t
        if st.last_bad is not None and t - st.last_bad > self.clear_s:  # episode over
            st.bad_since, st.last_bad, st.reported = None, None, False
        if status != MISSING:
            return False
        if st.bad_since is None:
            st.bad_since = t
        st.last_bad = t
        if not st.reported and t - st.bad_since >= self.confirm_s:
            st.reported = True
            return True
        return False

    def prune(self, t: float) -> None:
        """Forget tracks that have not been seen for a while."""
        for key in [k for k, s in self.state.items() if t - s.last_seen > self.prune_s]:
            del self.state[key]


class Rules:
    def __init__(self, items: Sequence[Item], confirm_seconds: float = 3.0,
                 clear_seconds: float = 2.0, min_person_height_px: float = 80.0):
        self.items = list(items)
        self.min_height = min_person_height_px
        self.timer = ViolationTimer(confirm_seconds, clear_seconds)

    @classmethod
    def from_config(cls, cfg: dict) -> "Rules":
        """Build from the `rules` section of config.yaml. Only items present there are checked."""
        items = []
        for name, event_type in EVENT_TYPES.items():
            if name in cfg:
                c = cfg[name]
                items.append(Item(name, event_type, tuple(c["present"]), tuple(c.get("absent", [])),
                                  tuple(c.get("region", (0.0, 1.0)))))
        return cls(items, cfg["confirm_seconds"], cfg["clear_seconds"], cfg["min_person_height_px"])

    def update(self, persons: Sequence[Track], ppe_dets: Sequence[Det], t: float) -> FrameResult:
        """Process one frame. persons are tracked people; ppe_dets are all non-person detections."""
        events: List[Event] = []
        statuses: Dict[int, Dict[str, Tuple[str, float]]] = {p.track_id: {} for p in persons}
        for item in self.items:
            assigned = assign_dets(persons, ppe_dets, item)
            for p in persons:
                status, conf = assess(p, assigned[p.track_id], item, self.min_height)
                statuses[p.track_id][item.name] = (status, conf)
                if self.timer.update((p.track_id, item.name), status, t):
                    events.append(Event(item.event_type, p.track_id, t, conf, p.box))
        self.timer.prune(t)
        return FrameResult(events, statuses)