from __future__ import annotations

import mimetypes
import os
import time
from collections import Counter
import json
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST
import yaml

from safety.configuration import (ConfigurationError, apply_camera_config, camera_config, validate_config,
                                  validate_zone_points)
from safety.storage import (REVIEW_STATUSES, connect, get_camera_config,
                            import_camera_configs, list_camera_configs, list_events,
                            list_vehicles, set_review, upsert_camera_config)


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
        configured_cameras = list_camera_configs(conn)
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
        "configured_cameras": configured_cameras,
    })


def _ensure_camera_configs(conn):
    if list_camera_configs(conn):
        return
    with open(settings.CONFIG_PATH, "r", encoding="utf-8") as stream:
        imported = validate_config(yaml.safe_load(stream))
    import_camera_configs(conn, imported)


def cameras(request):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        rows = [dict(row) for row in list_camera_configs(conn)]
    finally:
        conn.close()
    for row in rows:
        row["capabilities"] = ", ".join(json.loads(row["capabilities_json"]))
    return render(request, "dashboard/cameras.html", {"cameras": rows})


def camera_add(request):
    error = None
    values = {
        "camera_id": request.POST.get("camera_id", "").strip(),
        "name": request.POST.get("name", "").strip(),
        "source_type": request.POST.get("source_type", "recorded"),
        "source_path": request.POST.get("source_path", "").strip(),
        "capabilities": request.POST.get("capabilities", "ppe"),
    }
    if request.method == "POST":
        try:
            camera_id = values["camera_id"]
            if not camera_id or not camera_id.replace("_", "").replace("-", "").isalnum():
                raise ConfigurationError("Camera ID may contain only letters, numbers, '_' or '-'.")
            existing_conn = _db()
            try:
                if get_camera_config(existing_conn, camera_id) is not None:
                    raise ConfigurationError(f"Camera {camera_id!r} already exists.")
            finally:
                existing_conn.close()
            capabilities = frozenset(v.strip() for v in values["capabilities"].split(",") if v.strip())
            candidate = camera_config({
                "cameras": {camera_id: {
                    "name": values["name"] or camera_id,
                    "enabled": True,
                    "source": {"type": values["source_type"], "path": values["source_path"] or None},
                    "capabilities": list(capabilities),
                    "zones": {},
                    "rules": _default_rules(),
                }},
                "rules": _default_rules(),
            }, camera_id)
            conn = _db()
            try:
                upsert_camera_config(conn, candidate)
            finally:
                conn.close()
            return redirect("camera-edit", camera_id=camera_id)
        except (ConfigurationError, ValueError) as exc:
            error = str(exc)
    return render(request, "dashboard/camera_add.html", {"values": values, "error": error})


def camera_toggle(request, camera_id: str):
    if request.method != "POST":
        return HttpResponseBadRequest("Use POST to enable or disable a camera.")
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
        if camera is None:
            raise Http404
        updated = camera.__class__(
            id=camera.id, name=camera.name, enabled=not camera.enabled,
            source=camera.source, capabilities=camera.capabilities,
            zones=camera.zones, rules=camera.rules, vehicle=camera.vehicle,
        )
        upsert_camera_config(conn, updated)
    finally:
        conn.close()
    return redirect("cameras")


def camera_test(request, camera_id: str):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
    finally:
        conn.close()
    if camera is None:
        raise Http404
    result = _source_status(camera)
    return render(request, "dashboard/camera_test.html", {"camera": camera, "result": result})


def camera_replay(request, camera_id: str):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
    finally:
        conn.close()
    if camera is None:
        raise Http404
    return render(request, "dashboard/camera_commands.html", {
        "camera": camera,
        "source_ready": _source_status(camera)["ok"],
        "run_command": (
            f"python run.py --video {camera.source.path} --camera-id {camera.id} "
            "--annotate --device 0"
        ) if camera.source.path else "",
        "live_command": (
            f"python live.py --video {camera.source.path} --camera-id {camera.id} --device 0"
        ) if camera.source.path else "",
    })


def camera_live(request, camera_id: str):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
    finally:
        conn.close()
    if camera is None:
        raise Http404
    source = _recorded_source_path(camera)
    return render(request, "dashboard/camera_live.html", {
        "camera": camera,
        "source_ready": source is not None and camera.enabled,
    })


def camera_stream(request, camera_id: str):
    """Stream annotated inference frames to a browser using multipart JPEG."""
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
    finally:
        conn.close()
    if camera is None:
        raise Http404
    source = _recorded_source_path(camera)
    if not camera.enabled:
        return HttpResponseBadRequest("Camera is disabled.")
    if source is None:
        return HttpResponseBadRequest("Camera has no readable recorded source.")

    def frames():
        import cv2
        from live import LiveWall
        from run import build_detector, build_vehicle_models, detector_config, load_config
        from safety.storage import file_hash, start_run

        cfg = load_config(str(settings.CONFIG_PATH), settings.DB_PATH)
        cfg = apply_camera_config(cfg, camera)
        jobs = [(os.fspath(source), camera_id)]
        detector = build_detector(cfg)
        vehicle_models = build_vehicle_models(cfg, jobs)
        stream_conn = connect(settings.DB_PATH)
        run_id = start_run(
            stream_conn,
            file_hash(detector_config(cfg)["weights"]),
            file_hash(str(settings.CONFIG_PATH)),
            [source.name],
            f"dashboard live stream: {camera_id}",
        )
        wall = None
        try:
            wall = LiveWall(
                jobs, detector, cfg, stream_conn, run_id,
                tile_size=(960, 540), vehicle_models=vehicle_models,
            )
            start = time.monotonic()
            while not wall.all_done:
                image = wall.step(time.monotonic() - start)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("Could not encode an inference frame.")
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
                time.sleep(0.03)
        finally:
            if wall is not None:
                wall.close()
            stream_conn.close()

    response = StreamingHttpResponse(
        frames(), content_type="multipart/x-mixed-replace; boundary=frame"
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["X-Accel-Buffering"] = "no"
    return response


def camera_rules(request, camera_id: str):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
        if camera is None:
            raise Http404
        error = None
        if request.method == "POST":
            try:
                rules = _rules_from_form(request.POST)
                updated = camera.__class__(
                    id=camera.id, name=camera.name, enabled=camera.enabled,
                    source=camera.source, capabilities=camera.capabilities,
                    zones=camera.zones, rules=rules, vehicle=camera.vehicle,
                )
                upsert_camera_config(conn, updated)
                return redirect("camera-edit", camera_id=camera_id)
            except (ConfigurationError, ValueError) as exc:
                error = str(exc)
        return render(request, "dashboard/camera_rules.html", {
            "camera": camera, "rules": _rules_form_values(camera.rules), "error": error,
        })
    finally:
        conn.close()


def camera_video(request, camera_id: str):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
    finally:
        conn.close()
    if camera is None:
        raise Http404
    path = _recorded_source_path(camera)
    if path is None:
        raise Http404
    return FileResponse(path.open("rb"), content_type="video/mp4")


def _default_rules():
    return {
        "confirm_seconds": 3.0,
        "clear_seconds": 2.0,
        "min_person_height_px": 80,
        "helmet": {"present": ["Hardhat"], "absent": ["NO-Hardhat"], "region": [0.0, 0.4]},
        "vest": {"present": ["Safety-Vest"], "absent": ["NO-Safety-Vest"], "region": [0.15, 0.75]},
    }


def _rules_form_values(rules):
    values = {
        "confirm_seconds": rules.get("confirm_seconds", 3.0),
        "clear_seconds": rules.get("clear_seconds", 2.0),
        "min_person_height_px": rules.get("min_person_height_px", 80),
    }
    for name in ("helmet", "vest"):
        item = rules.get(name, {})
        values[name] = {
            "enabled": name in rules,
            "present": ", ".join(item.get("present", [])),
            "absent": ", ".join(item.get("absent", [])),
            "region_start": item.get("region", [0.0, 1.0])[0],
            "region_end": item.get("region", [0.0, 1.0])[1],
        }
    return values


def _rules_from_form(data):
    def number(name, minimum=0):
        try:
            value = float(data.get(name, ""))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name} must be a number.") from exc
        if value < minimum:
            raise ConfigurationError(f"{name} must be at least {minimum}.")
        return value

    rules = {
        "confirm_seconds": number("confirm_seconds"),
        "clear_seconds": number("clear_seconds"),
        "min_person_height_px": number("min_person_height_px"),
    }
    for name in ("helmet", "vest"):
        if data.get(f"{name}_enabled") != "on":
            continue
        present = [v.strip() for v in data.get(f"{name}_present", "").split(",") if v.strip()]
        absent = [v.strip() for v in data.get(f"{name}_absent", "").split(",") if v.strip()]
        if not present:
            raise ConfigurationError(f"{name} needs at least one present class.")
        start = number(f"{name}_region_start")
        end = number(f"{name}_region_end")
        if end > 1 or start > 1 or start >= end:
            raise ConfigurationError(f"{name} region must be within 0..1 and start before end.")
        rules[name] = {"present": present, "absent": absent, "region": [start, end]}
    return rules


def _recorded_source_path(camera):
    if camera.source.type != "recorded" or not camera.source.path:
        return None
    source = Path(camera.source.path)
    if not source.is_absolute():
        source = settings.ROOT_DIR / source
    source = source.resolve()
    return source if source.is_file() else None


def _source_status(camera):
    path = _recorded_source_path(camera)
    if camera.source.type != "recorded":
        return {"ok": False, "message": "RTSP source testing is not available in the recorded-video dashboard yet."}
    if path is None:
        return {"ok": False, "message": "Recorded video file was not found."}
    import cv2
    cap = cv2.VideoCapture(os.fspath(path))
    try:
        if not cap.isOpened():
            return {"ok": False, "message": "OpenCV could not open the video file."}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {"ok": True, "message": f"Source is ready: {width}x{height}, {fps:.2f} FPS, {frames} frames."}
    finally:
        cap.release()


def camera_edit(request, camera_id: str):
    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
        if camera is None:
            raise Http404
        error = None
        if request.method == "POST":
            try:
                action = request.POST.get("action", "save_camera")
                if action in {"save_zone", "delete_zone"}:
                    zones = dict(camera.zones)
                    zone_name = request.POST.get("zone_name", "").strip()
                    if not zone_name or zone_name not in {"detection_zone", "gate_zone"}:
                        raise ConfigurationError("Choose a valid zone name.")
                    if action == "delete_zone":
                        zones.pop(zone_name, None)
                    else:
                        polygon = _parse_zone_points(request.POST.get("points", ""))
                        validate_zone_points(polygon, f"zones.{zone_name}")
                        zones[zone_name] = polygon
                    updated = camera.__class__(
                        id=camera.id, name=camera.name, enabled=camera.enabled,
                        source=camera.source, capabilities=camera.capabilities,
                        zones=zones, rules=camera.rules, vehicle=camera.vehicle,
                    )
                    upsert_camera_config(conn, updated)
                    return redirect(f"/cameras/{camera_id}/?zone={zone_name}")
                capabilities = [v.strip() for v in request.POST.get("capabilities", "").split(",") if v.strip()]
                rules = json.loads(request.POST.get("rules", "{}"))
                vehicle_raw = request.POST.get("vehicle", "").strip()
                vehicle = json.loads(vehicle_raw) if vehicle_raw else None
                candidate = camera.__class__(
                    id=camera.id,
                    name=request.POST.get("name", "").strip() or camera.id,
                    enabled=request.POST.get("enabled") == "on",
                    source=camera.source.__class__(
                        request.POST.get("source_type", camera.source.type),
                        request.POST.get("source_path", "").strip() or None,
                    ),
                    capabilities=frozenset(capabilities),
                    zones=camera.zones,
                    rules=rules,
                    vehicle=vehicle,
                )
                normalized = camera_config({
                    "cameras": {camera_id: {
                        "name": candidate.name, "enabled": candidate.enabled,
                        "source": {"type": candidate.source.type, "path": candidate.source.path},
                        "capabilities": list(candidate.capabilities),
                        "zones": candidate.zones, "rules": candidate.rules,
                        "vehicle": candidate.vehicle,
                    }},
                    "gate": {"camera_id": camera_id, **(candidate.vehicle or {})}
                    if candidate.vehicle else {},
                }, camera_id)
                upsert_camera_config(conn, normalized)
                return redirect("cameras")
            except (ConfigurationError, json.JSONDecodeError, TypeError, ValueError) as exc:
                error = str(exc)
        selected_zone = request.GET.get("zone", "detection_zone")
        return render(request, "dashboard/camera_edit.html", {
            "camera": camera,
            "error": error,
            "zone_names": ("detection_zone", "gate_zone"),
            "selected_zone": selected_zone,
            "zone_points": _format_zone_points(camera.zones.get(selected_zone, [])),
            "zone_points_json": json.dumps(camera.zones.get(selected_zone, [])),
            "has_recorded_source": camera.source.type == "recorded" and bool(camera.source.path),
            "rules": json.dumps(camera.rules, indent=2),
            "vehicle": json.dumps(camera.vehicle, indent=2) if camera.vehicle else "",
            "capabilities": ", ".join(sorted(camera.capabilities)),
        })
    finally:
        conn.close()


def camera_frame(request, camera_id: str):
    """Return a JPEG frame from a camera's assigned recorded source."""
    import cv2

    conn = _db()
    try:
        _ensure_camera_configs(conn)
        camera = get_camera_config(conn, camera_id)
    finally:
        conn.close()
    if camera is None:
        raise Http404
    if camera.source.type != "recorded" or not camera.source.path:
        return HttpResponseBadRequest("Zone preview requires a recorded video source.")

    source = Path(camera.source.path)
    if not source.is_absolute():
        source = (settings.ROOT_DIR / source).resolve()
    else:
        source = source.resolve()
    if not source.is_file():
        raise Http404

    cap = cv2.VideoCapture(os.fspath(source))
    try:
        if not cap.isOpened():
            raise Http404
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            time_s = max(0.0, float(request.GET.get("time", "0")))
        except ValueError:
            return HttpResponseBadRequest("time must be a number.")
        frame_index = max(0, round(time_s * fps))
        if frame_count:
            frame_index = min(frame_index, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise Http404
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return HttpResponse("Could not encode video frame.", status=500)
        response = HttpResponse(encoded.tobytes(), content_type="image/jpeg")
        response["X-Frame-Time"] = f"{frame_index / fps:.3f}"
        return response
    finally:
        cap.release()


def _parse_zone_points(value: str):
    points = []
    for line_number, line in enumerate(value.splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            raise ConfigurationError(f"Point line {line_number} must be formatted as x,y")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ConfigurationError(f"Point line {line_number} must contain numbers") from exc
    return points


def _format_zone_points(points):
    return "\n".join(f"{point[0]:.6f},{point[1]:.6f}" for point in points)


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
