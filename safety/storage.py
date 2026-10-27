"""Layer 5: SQLite storage for runs and events. Plain functions, no ORM.

Every event belongs to a run; the run records which model and config produced it,
so results can be compared and reproduced.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .types import Event, VehicleRecord
from .configuration import CameraConfig, CameraSource

REVIEW_STATUSES = ("unreviewed", "confirmed", "false_positive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    config_version TEXT NOT NULL,
    clips          TEXT NOT NULL,              -- JSON list of clip names
    notes          TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    event_type     TEXT NOT NULL,
    camera_id      TEXT NOT NULL,
    clip           TEXT NOT NULL,
    video_time_s   REAL NOT NULL,
    timestamp_utc  TEXT NOT NULL,              -- clip start + video_time_s
    track_id       INTEGER NOT NULL,
    confidence     REAL NOT NULL,
    bbox_json      TEXT NOT NULL,
    snapshot_path  TEXT NOT NULL,
    review_status  TEXT NOT NULL DEFAULT 'unreviewed',
    reviewed_at    TEXT,
    review_notes   TEXT NOT NULL DEFAULT '',
    detail         TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS vehicle_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    camera_id      TEXT NOT NULL,
    clip           TEXT NOT NULL,
    timestamp_utc  TEXT NOT NULL,              -- clip start + first_seen_s
    first_seen_s   REAL NOT NULL,
    last_seen_s    REAL NOT NULL,
    track_id       INTEGER NOT NULL,
    vehicle_class  TEXT NOT NULL,
    class_conf     REAL NOT NULL,
    plate          TEXT,                       -- plate text, a hash, or NULL (see store_plate in config)
    plate_conf     REAL NOT NULL DEFAULT 0,
    snapshot_path  TEXT NOT NULL DEFAULT '',
    bbox_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vehicle_run ON vehicle_log (run_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events (event_type, timestamp_utc);
CREATE TABLE IF NOT EXISTS camera_configs (
    camera_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    source_type     TEXT NOT NULL,
    source_path     TEXT,
    capabilities_json TEXT NOT NULL,
    zones_json      TEXT NOT NULL,
    rules_json      TEXT NOT NULL,
    vehicle_json    TEXT,
    revision        INTEGER NOT NULL DEFAULT 1,
    updated_at      TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open the database, creating the file and tables if needed."""
    folder = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
    if "detail" not in columns:   # database created before this column existed
        conn.execute("ALTER TABLE events ADD COLUMN detail TEXT NOT NULL DEFAULT ''")
        conn.commit()
    return conn


def file_hash(path: str) -> str:
    """Short fingerprint of a file (config or weights), used as a version label."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def start_run(conn: sqlite3.Connection, model_version: str, config_version: str,
              clips: List[str], notes: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO runs (started_at, model_version, config_version, clips, notes) VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), model_version, config_version,
         json.dumps(clips), notes),
    )
    conn.commit()
    return int(cur.lastrowid)


def latest_run_id(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute("SELECT MAX(id) AS id FROM runs").fetchone()
    return row["id"]


def upsert_camera_config(conn: sqlite3.Connection, camera: CameraConfig) -> int:
    """Store the latest dashboard-managed camera configuration."""
    current = conn.execute(
        "SELECT revision FROM camera_configs WHERE camera_id = ?", (camera.id,)
    ).fetchone()
    revision = int(current["revision"]) + 1 if current else 1
    conn.execute(
        "INSERT INTO camera_configs "
        "(camera_id, name, enabled, source_type, source_path, capabilities_json, "
        "zones_json, rules_json, vehicle_json, revision, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name, enabled=excluded.enabled, "
        "source_type=excluded.source_type, source_path=excluded.source_path, "
        "capabilities_json=excluded.capabilities_json, zones_json=excluded.zones_json, "
        "rules_json=excluded.rules_json, vehicle_json=excluded.vehicle_json, "
        "revision=excluded.revision, updated_at=excluded.updated_at",
        (camera.id, camera.name, int(camera.enabled), camera.source.type, camera.source.path,
         json.dumps(sorted(camera.capabilities)), json.dumps(camera.zones, sort_keys=True),
         json.dumps(camera.rules, sort_keys=True),
         json.dumps(camera.vehicle, sort_keys=True) if camera.vehicle is not None else None,
         revision, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return revision


def list_camera_configs(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return dashboard-managed cameras ordered by name."""
    return conn.execute("SELECT * FROM camera_configs ORDER BY name, camera_id").fetchall()


def get_camera_config(conn: sqlite3.Connection, camera_id: str) -> Optional[CameraConfig]:
    """Load one dashboard-managed camera, or None when it has not been imported."""
    row = conn.execute("SELECT * FROM camera_configs WHERE camera_id = ?", (camera_id,)).fetchone()
    if row is None:
        return None
    return CameraConfig(
        id=row["camera_id"],
        name=row["name"],
        enabled=bool(row["enabled"]),
        source=CameraSource(row["source_type"], row["source_path"]),
        capabilities=frozenset(json.loads(row["capabilities_json"])),
        zones=json.loads(row["zones_json"]),
        rules=json.loads(row["rules_json"]),
        vehicle=json.loads(row["vehicle_json"]) if row["vehicle_json"] else None,
    )


def import_camera_configs(conn: sqlite3.Connection, cameras: dict[str, CameraConfig]) -> int:
    """Import normalized YAML cameras without overwriting unchanged data semantics."""
    for camera in cameras.values():
        upsert_camera_config(conn, camera)
    return len(cameras)


def add_event(conn: sqlite3.Connection, run_id: int, event: Event, camera_id: str,
              clip: str, clip_start: datetime, snapshot_path: str) -> int:
    if clip_start.tzinfo is None:
        raise ValueError("clip_start must include a timezone, e.g. 2026-10-19T08:30:00+10:00")
    timestamp = (clip_start + timedelta(seconds=event.video_time_s)).astimezone(timezone.utc)
    cur = conn.execute(
        "INSERT INTO events (run_id, event_type, camera_id, clip, video_time_s, timestamp_utc,"
        " track_id, confidence, bbox_json, snapshot_path, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, event.event_type, camera_id, clip, round(event.video_time_s, 3),
         timestamp.isoformat(), event.track_id, round(event.confidence, 4),
         json.dumps([round(v, 1) for v in event.box]), snapshot_path, event.detail),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_events(conn: sqlite3.Connection, run_id: Optional[int] = None,
                event_type: Optional[str] = None, camera_id: Optional[str] = None,
                review_status: Optional[str] = None) -> List[sqlite3.Row]:
    """Events, oldest first. Any filter left as None is ignored."""
    where, args = [], []
    for column, value in (("run_id", run_id), ("event_type", event_type),
                          ("camera_id", camera_id), ("review_status", review_status)):
        if value is not None:
            where.append(f"{column} = ?")
            args.append(value)
    sql = "SELECT * FROM events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp_utc, id"
    return conn.execute(sql, args).fetchall()


def set_review(conn: sqlite3.Connection, event_id: int, status: str, notes: str = "") -> None:
    """Mark an event confirmed or a false positive (the dashboard's review buttons call this)."""
    if status not in REVIEW_STATUSES:
        raise ValueError(f"status must be one of {REVIEW_STATUSES}, got {status!r}")
    cur = conn.execute(
        "UPDATE events SET review_status = ?, reviewed_at = ?, review_notes = ? WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(), notes, event_id),
    )
    if cur.rowcount == 0:
        raise ValueError(f"No event with id {event_id}")
    conn.commit()


def add_vehicle(conn: sqlite3.Connection, run_id: int, rec: VehicleRecord, camera_id: str, clip: str,
                clip_start: datetime, plate: Optional[str], snapshot_path: str = "") -> int:
    """Log one vehicle. `plate` is what should be stored (text, hash or None), already decided by the caller."""
    if clip_start.tzinfo is None:
        raise ValueError("clip_start must include a timezone, e.g. 2026-10-19T08:30:00+10:00")
    seen = (clip_start + timedelta(seconds=rec.first_seen_s)).astimezone(timezone.utc)
    cur = conn.execute(
        "INSERT INTO vehicle_log (run_id, camera_id, clip, timestamp_utc, first_seen_s, last_seen_s,"
        " track_id, vehicle_class, class_conf, plate, plate_conf, snapshot_path, bbox_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, camera_id, clip, seen.isoformat(), round(rec.first_seen_s, 3), round(rec.last_seen_s, 3),
         rec.track_id, rec.vehicle_class, round(rec.class_conf, 4), plate, round(rec.plate_conf, 4),
         snapshot_path, json.dumps([round(v, 1) for v in rec.box])),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_vehicles(conn: sqlite3.Connection, run_id: Optional[int] = None, camera_id: Optional[str] = None,
                  vehicle_class: Optional[str] = None, plate: Optional[str] = None) -> List[sqlite3.Row]:
    """Logged vehicles, oldest first. Any filter left as None is ignored."""
    where, args = [], []
    for column, value in (("run_id", run_id), ("camera_id", camera_id),
                          ("vehicle_class", vehicle_class), ("plate", plate)):
        if value is not None:
            where.append(f"{column} = ?")
            args.append(value)
    sql = "SELECT * FROM vehicle_log"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp_utc, id"
    return conn.execute(sql, args).fetchall()
