"""Normalized camera configuration shared by pipeline entry points.

The initial configuration remains YAML-backed. This module gives callers one
camera-centric view while accepting the existing configuration shape during the
migration to dashboard-managed configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple


class ConfigurationError(ValueError):
    """Raised when camera configuration cannot be used safely."""


def validate_zone_points(points: Any, field: str = "zone") -> Tuple[Tuple[float, float], ...]:
    """Validate a normalized polygon and return immutable numeric points."""
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ConfigurationError(f"{field} must be a list of points")
    if len(points) < 3:
        raise ConfigurationError(f"{field} needs at least 3 points")
    normalized = []
    for index, point in enumerate(points):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ConfigurationError(f"{field} point {index + 1} must contain exactly x and y")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{field} point {index + 1} must contain numbers") from exc
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ConfigurationError(f"{field} point {index + 1} must be between 0 and 1")
        normalized.append((x, y))
    return tuple(normalized)


@dataclass(frozen=True)
class CameraSource:
    type: str = "recorded"
    path: Optional[str] = None


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    enabled: bool
    source: CameraSource
    capabilities: FrozenSet[str]
    zones: Mapping[str, Any]
    rules: Mapping[str, Any]
    vehicle: Optional[Mapping[str, Any]]

    @property
    def supports_ppe(self) -> bool:
        return "ppe" in self.capabilities

    @property
    def supports_vehicles(self) -> bool:
        return "vehicles" in self.capabilities

    @property
    def supports_licence_plates(self) -> bool:
        return "licence_plates" in self.capabilities


def _camera_specs(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    cameras = cfg.get("cameras") or {}
    if not isinstance(cameras, Mapping):
        raise ConfigurationError("cameras must be a mapping keyed by camera ID")
    return cameras


def camera_config(cfg: Mapping[str, Any], camera_id: str) -> CameraConfig:
    """Return the normalized configuration for one camera.

    Legacy YAML is supported: an omitted camera entry means default PPE
    processing, and the legacy ``gate.camera_id`` assignment enables vehicle
    processing for that camera.
    """
    if not camera_id or not isinstance(camera_id, str):
        raise ConfigurationError("camera_id must be a non-empty string")

    spec = dict(_camera_specs(cfg).get(camera_id) or {})
    source_spec = spec.get("source") or {}
    if not isinstance(source_spec, Mapping):
        raise ConfigurationError(f"cameras.{camera_id}.source must be a mapping")
    source_type = source_spec.get("type", "recorded")
    if source_type not in {"recorded", "rtsp"}:
        raise ConfigurationError(
            f"cameras.{camera_id}.source.type must be 'recorded' or 'rtsp', got {source_type!r}"
        )

    gate = cfg.get("gate") or {}
    legacy_gate = gate.get("camera_id") == camera_id
    configured_capabilities = spec.get("capabilities")
    if configured_capabilities is None:
        capabilities = {"ppe"}
        if legacy_gate:
            capabilities.add("vehicles")
            if gate.get("read_plates", False):
                capabilities.add("licence_plates")
    else:
        if not isinstance(configured_capabilities, (list, tuple, set)):
            raise ConfigurationError(f"cameras.{camera_id}.capabilities must be a list")
        capabilities = set(configured_capabilities)
        unknown = capabilities - {"ppe", "vehicles", "licence_plates"}
        if unknown:
            raise ConfigurationError(
                f"cameras.{camera_id}.capabilities contains unsupported values: {sorted(unknown)}"
            )

    vehicle = spec.get("vehicle")
    if vehicle is None and legacy_gate:
        vehicle = gate
    if "vehicles" in capabilities and vehicle is None:
        vehicle = spec.get("vehicle")
        if vehicle is None:
            raise ConfigurationError(
                f"cameras.{camera_id} enables vehicles but has no vehicle configuration"
            )
    if "licence_plates" in capabilities and not (vehicle or {}).get("read_plates", False):
        raise ConfigurationError(
            f"cameras.{camera_id} enables licence_plates but vehicle.read_plates is false"
        )

    rules = spec.get("rules")
    if rules is None:
        rules = cfg.get("rules") or {}
    if not isinstance(rules, Mapping):
        raise ConfigurationError(f"cameras.{camera_id}.rules must be a mapping")

    zones = spec.get("zones")
    if zones is None:
        zones = {key: spec[key] for key in ("detection_zone", "gate_zone") if key in spec}
    if not isinstance(zones, Mapping):
        raise ConfigurationError(f"cameras.{camera_id}.zones must be a mapping")
    for zone_name, points in zones.items():
        if points is not None:
            validate_zone_points(points, f"cameras.{camera_id}.zones.{zone_name}")

    return CameraConfig(
        id=camera_id,
        name=str(spec.get("name") or camera_id),
        enabled=bool(spec.get("enabled", True)),
        source=CameraSource(source_type, source_spec.get("path")),
        capabilities=frozenset(capabilities),
        zones=zones,
        rules=rules,
        vehicle=vehicle,
    )


def validate_config(cfg: Mapping[str, Any]) -> Dict[str, CameraConfig]:
    """Validate and normalize every declared camera plus the legacy gate camera."""
    cameras = dict(_camera_specs(cfg))
    gate = cfg.get("gate") or {}
    gate_id = gate.get("camera_id")
    if gate_id:
        cameras.setdefault(gate_id, {})
    normalized = {camera_id: camera_config(cfg, camera_id) for camera_id in cameras}
    return normalized


def as_legacy_config(cfg: Mapping[str, Any], camera: CameraConfig) -> dict:
    """Build the small legacy-shaped view required by existing layer helpers."""
    output = dict(cfg)
    output["rules"] = dict(camera.rules)
    output["cameras"] = {
        camera.id: {
            "detection_zone": camera.zones.get("detection_zone"),
            "gate_zone": camera.zones.get("gate_zone"),
        }
    }
    if camera.vehicle is not None:
        output["gate"] = dict(camera.vehicle)
    else:
        output.pop("gate", None)
    return output


def apply_camera_config(cfg: Mapping[str, Any], camera: CameraConfig) -> dict:
    """Overlay one persisted camera config onto global model settings.

    Global model/tracking settings remain YAML-backed for now, while all
    camera-owned settings come from the dashboard/database without relying on
    legacy ``gate`` or YAML camera entries.
    """
    output = dict(cfg)
    cameras = dict(output.get("cameras") or {})
    cameras[camera.id] = {
        "name": camera.name,
        "enabled": camera.enabled,
        "source": {"type": camera.source.type, "path": camera.source.path},
        "capabilities": list(camera.capabilities),
        "zones": dict(camera.zones),
        "rules": dict(camera.rules),
    }
    if camera.vehicle is not None:
        cameras[camera.id]["vehicle"] = dict(camera.vehicle)
    output["cameras"] = cameras
    return output
