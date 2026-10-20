"""Score a pipeline run against hand-labelled events.

Usage:
    python evaluate.py                       # scores the latest run
    python evaluate.py --run-id 3 --tolerance 2.0

labels.csv columns: clip,camera_id,event_type,start_s,end_s,condition
  - one row per true violation; start_s and end_s are video seconds within that clip
  - a clip with no violations gets one row with event_type "none" (start_s, end_s blank)
  - condition is free text used for breakdowns (day, night, far ...); optional
  - only clips listed in labels.csv are scored

An event counts as correct when its type and camera match a label and its confirmation time
falls between start_s - tolerance and end_s + tolerance. Each label can be matched once; extra
events are false alarms (and counted as duplicates when they fall inside a label's window).
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

from safety.storage import connect, latest_run_id, list_events

REQUIRED_COLUMNS = {"clip", "camera_id", "event_type", "start_s", "end_s"}


def read_labels(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for line, r in enumerate(reader, start=2):
            row = {
                "clip": r["clip"].strip(),
                "camera_id": (r["camera_id"] or "").strip(),
                "event_type": (r["event_type"] or "").strip(),
                "condition": (r.get("condition") or "").strip(),
            }
            if row["event_type"] != "none":
                try:
                    row["start_s"], row["end_s"] = float(r["start_s"]), float(r["end_s"])
                except (TypeError, ValueError):
                    raise ValueError(f"{path} line {line}: start_s and end_s must be numbers")
                if row["end_s"] < row["start_s"]:
                    raise ValueError(f"{path} line {line}: end_s is before start_s")
            rows.append(row)
    return rows


def in_window(label: dict, event: dict, tolerance: float) -> bool:
    return (label["event_type"] == event["event_type"]
            and label["camera_id"] == event["camera_id"]
            and label["start_s"] - tolerance <= event["video_time_s"] <= label["end_s"] + tolerance)


def score_clip(labels: List[dict], events: List[dict], tolerance: float) -> dict:
    """Score one clip. labels must exclude 'none' rows. Returns counts and confirmation delays."""
    free = list(labels)
    matched, extra = [], []
    for ev in sorted(events, key=lambda e: e["video_time_s"]):
        hit = next((lab for lab in free if in_window(lab, ev, tolerance)), None)
        if hit is None:
            extra.append(ev)
        else:
            free.remove(hit)
            matched.append((hit, ev))
    duplicates = sum(1 for ev in extra if any(in_window(lab, ev, tolerance) for lab in labels))
    return {
        "tp": len(matched),
        "fn": len(free),
        "fp": len(extra),
        "dup": duplicates,
        "delays": [ev["video_time_s"] - lab["start_s"] for lab, ev in matched],
        "matched": matched,   # (label, event) pairs
        "missed": free,       # labels with no event
        "extra": extra,       # events with no label
    }


def video_duration_s(path: str) -> Optional[float]:
    """Clip length in seconds, or None if the file or OpenCV is not available."""
    try:
        import cv2
    except ImportError:
        return None
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps and frames else None


def print_details(clip: str, result: dict) -> None:
    """One line per label (HIT or MISS) and per false alarm, so each number can be traced."""
    print(f"\n{clip}")
    for lab, ev in result["matched"]:
        print(f"  HIT    {lab['event_type']:<10} labelled {lab['start_s']:.1f}-{lab['end_s']:.1f}s"
              f"  -> event at {ev['video_time_s']:.1f}s (conf {ev['confidence']:.2f}, track {ev['track_id']})")
    for lab in result["missed"]:
        print(f"  MISS   {lab['event_type']:<10} labelled {lab['start_s']:.1f}-{lab['end_s']:.1f}s")
    for ev in result["extra"]:
        print(f"  FALSE  {ev['event_type']:<10} event at {ev['video_time_s']:.1f}s "
              f"(conf {ev['confidence']:.2f}, track {ev['track_id']})")


def ratio(num: float, den: float) -> str:
    return f"{num / den:.2f}" if den else "n/a"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--db", default="outputs/events.db")
    ap.add_argument("--clips-dir", default="data/clips", help="used only to measure clip length")
    ap.add_argument("--run-id", type=int, help="default: the latest run")
    ap.add_argument("--tolerance", type=float, default=2.0, help="seconds either side of a label")
    ap.add_argument("--details", action="store_true", help="list every HIT, MISS and FALSE alarm")
    args = ap.parse_args(argv)

    labels = read_labels(args.labels)
    conn = connect(args.db)
    run_id = args.run_id or latest_run_id(conn)
    if run_id is None:
        raise SystemExit("No runs in the database yet. Run the pipeline first.")
    events = [dict(r) for r in list_events(conn, run_id=run_id)]

    groups: Dict[str, dict] = {}
    details = []
    for clip in sorted({lab["clip"] for lab in labels}):
        clip_rows = [lab for lab in labels if lab["clip"] == clip]
        condition = next((r["condition"] for r in clip_rows if r["condition"]), "all")
        result = score_clip([r for r in clip_rows if r["event_type"] != "none"],
                            [e for e in events if e["clip"] == clip], args.tolerance)
        details.append((clip, result))
        seconds = video_duration_s(os.path.join(args.clips_dir, clip))
        for name in (condition, "TOTAL"):
            g = groups.setdefault(name, {"clips": 0, "hours": 0.0, "no_length": 0,
                                         "tp": 0, "fn": 0, "fp": 0, "dup": 0, "delays": []})
            g["clips"] += 1
            g["tp"] += result["tp"]
            g["fn"] += result["fn"]
            g["fp"] += result["fp"]
            g["dup"] += result["dup"]
            g["delays"] += result["delays"]
            if seconds is None:
                g["no_length"] += 1
            else:
                g["hours"] += seconds / 3600

    print(f"Run {run_id} | tolerance {args.tolerance}s | {len(events)} events scored against {args.labels}")
    print(f"{'group':<10}{'clips':>6}{'TP':>5}{'FN':>5}{'FP':>5}{'dup':>5}"
          f"{'precision':>11}{'recall':>8}{'FA/hour':>9}{'delay_s':>9}")
    for name in sorted(groups, key=lambda n: (n == "TOTAL", n)):
        g = groups[name]
        fa_per_hour = ratio(g["fp"], g["hours"]) if g["no_length"] == 0 else "n/a"
        delay = f"{sum(g['delays']) / len(g['delays']):.1f}" if g["delays"] else "n/a"
        print(f"{name:<10}{g['clips']:>6}{g['tp']:>5}{g['fn']:>5}{g['fp']:>5}{g['dup']:>5}"
              f"{ratio(g['tp'], g['tp'] + g['fp']):>11}{ratio(g['tp'], g['tp'] + g['fn']):>8}"
              f"{fa_per_hour:>9}{delay:>9}")
    if args.details:
        for clip, result in details:
            print_details(clip, result)
    if groups["TOTAL"]["no_length"]:
        print(f"Note: FA/hour is n/a because {groups['TOTAL']['no_length']} clip(s) were not found in "
              f"{args.clips_dir} (or OpenCV is missing).")


if __name__ == "__main__":
    main()