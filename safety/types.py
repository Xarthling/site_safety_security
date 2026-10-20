"""Shared data types. Every layer imports from here; this file imports nothing from the project."""
from dataclasses import dataclass
from typing import Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

# PPE status of one item (helmet, vest) on one person
OK, MISSING, UNKNOWN = "ok", "missing", "unknown"

# Event types written to the database
NO_HELMET, NO_VEST, VEHICLE_IN_ZONE = "no_helmet", "no_vest", "vehicle_in_zone"


@dataclass(frozen=True)
class Det:
    """One raw detection from a model."""
    cls: str       # class name as the model reports it, e.g. "helmet"
    conf: float
    box: Box


@dataclass(frozen=True)
class Track:
    """A detection with a stable ID from the tracker. label is 'person' or 'vehicle'."""
    track_id: int
    label: str
    conf: float
    box: Box


@dataclass(frozen=True)
class Event:
    """A confirmed violation, produced by the rules layer."""
    event_type: str      # NO_HELMET, NO_VEST or VEHICLE_IN_ZONE
    track_id: int
    video_time_s: float  # seconds from the start of the clip, when the event was confirmed
    confidence: float
    box: Box


@dataclass(frozen=True)
class VehicleRecord:
    """One vehicle seen at the gate, logged when it leaves view."""
    track_id: int
    vehicle_class: str           # most likely class over the vehicle's whole pass, e.g. "car"
    class_conf: float            # average detection confidence for that class
    plate_text: Optional[str]    # normalised plate (letters and digits), None if none was read
    plate_conf: float            # confidence of that plate read, 0.0 if none
    first_seen_s: float          # video seconds
    last_seen_s: float
    box: Box                     # where it was last seen
