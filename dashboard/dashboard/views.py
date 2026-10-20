from __future__ import annotations

import mimetypes
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from safety.storage import connect, list_events, list_vehicles, set_review


def _db():
    return connect(settings.DB_PATH)


def index(request):
    run_id = request.GET.get("run_id") or None
    event_type = request.GET.get("event_type") or None
    camera_id = request.GET.get("camera_id") or None
    review_status = request.GET.get("review_status") or None
    conn = _db()
    try:
        events = []
        for row in list_events(conn, run_id=run_id, event_type=event_type,
                               camera_id=camera_id, review_status=review_status):
            event = dict(row)
            event["snapshot_name"] = _snapshot_name(event["snapshot_path"])
            events.append(event)
        vehicles = list_vehicles(conn, run_id=run_id, camera_id=camera_id)
        runs = conn.execute("SELECT id, started_at, clips, notes FROM runs ORDER BY id DESC").fetchall()
    finally:
        conn.close()
    return render(request, "dashboard/index.html", {
        "events": events,
        "vehicles": vehicles,
        "runs": runs,
        "selected": {"run_id": run_id, "event_type": event_type,
                     "camera_id": camera_id, "review_status": review_status},
        "event_count": len(events),
        "vehicle_count": len(vehicles),
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
