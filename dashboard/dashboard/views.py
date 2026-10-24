from __future__ import annotations

import mimetypes
from collections import Counter
import json
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from safety.storage import REVIEW_STATUSES, connect, list_events, list_vehicles, set_review


def _db():
    return connect(settings.DB_PATH)


def index(request):
    run_id = request.GET.get("run_id") or None
    event_type = request.GET.get("event_type") or None
    camera_id = request.GET.get("camera_id") or None
    review_status = request.GET.get("review_status") or None
    vehicle_class = request.GET.get("vehicle_class") or None
    conn = _db()
    try:
        events = []
        for row in list_events(conn, run_id=run_id, event_type=event_type,
                               camera_id=camera_id, review_status=review_status):
            event = dict(row)
            event["snapshot_name"] = _snapshot_name(event["snapshot_path"])
            events.append(event)
        vehicles = []
        for row in list_vehicles(conn, run_id=run_id, camera_id=camera_id,
                                 vehicle_class=vehicle_class):
            vehicle = dict(row)
            vehicle["snapshot_name"] = _snapshot_name(vehicle["snapshot_path"])
            vehicles.append(vehicle)
        runs = [dict(row) for row in conn.execute(
            "SELECT id, started_at, model_version, config_version, clips, notes "
            "FROM runs ORDER BY id DESC"
        ).fetchall()]
        for run in runs:
            try:
                run["clips"] = json.loads(run["clips"])
            except (TypeError, json.JSONDecodeError):
                run["clips"] = []
        event_types = [row["event_type"] for row in conn.execute(
            "SELECT DISTINCT event_type FROM events ORDER BY event_type"
        ).fetchall()]
        cameras = [row["camera_id"] for row in conn.execute(
            "SELECT camera_id FROM (SELECT camera_id FROM events UNION SELECT camera_id FROM vehicle_log) "
            "ORDER BY camera_id"
        ).fetchall()]
        vehicle_classes = [row["vehicle_class"] for row in conn.execute(
            "SELECT DISTINCT vehicle_class FROM vehicle_log ORDER BY vehicle_class"
        ).fetchall()]
    finally:
        conn.close()
    event_review_counts = Counter(event["review_status"] for event in events)
    return render(request, "dashboard/index.html", {
        "events": events,
        "vehicles": vehicles,
        "runs": runs,
        "selected": {"run_id": run_id, "event_type": event_type,
                     "camera_id": camera_id, "review_status": review_status,
                     "vehicle_class": vehicle_class},
        "event_count": len(events),
        "vehicle_count": len(vehicles),
        "event_review_counts": event_review_counts,
        "review_counts": {status: event_review_counts.get(status, 0) for status in REVIEW_STATUSES},
        "event_types": event_types,
        "cameras": cameras,
        "vehicle_classes": vehicle_classes,
        "review_statuses": ("unreviewed", "confirmed", "false_positive"),
    })


@require_POST
def review_event(request, event_id: int):
    status = request.POST.get("status", "")
    if status not in {"confirmed", "false_positive", "unreviewed"}:
        return HttpResponseBadRequest("Invalid review status.")
    conn = _db()
    try:
        set_review(conn, event_id, status, request.POST.get("notes", "").strip())
    finally:
        conn.close()
    return redirect(request.POST.get("next") or "dashboard")


def snapshot(request, relative_path: str):
    root = settings.SNAPSHOT_ROOT
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise Http404
    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return FileResponse(candidate.open("rb"), content_type=content_type)


def _snapshot_name(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(settings.SNAPSHOT_ROOT))
    except ValueError:
        return ""
