import pytest
import yaml

from run import load_config
from safety.configuration import ConfigurationError, camera_config, validate_config


def test_existing_yaml_normalizes_gate_camera():
    cfg = load_config("config.yaml")
    camera = camera_config(cfg, "gate_cam")

    assert camera.name == "Main gate"
    assert camera.enabled
    assert camera.capabilities == frozenset({"ppe", "vehicles", "licence_plates"})
    assert camera.supports_vehicles
    assert camera.supports_licence_plates
    assert camera.zones["gate_zone"]
    assert camera.vehicle["camera_id"] == "gate_cam"


def test_legacy_unlisted_camera_keeps_ppe_defaults():
    cfg = load_config("config.yaml")
    camera = camera_config(cfg, "site_cam")

    assert camera.capabilities == frozenset({"ppe"})
    assert camera.name == "site_cam"
    assert camera.vehicle is None


def test_validate_config_includes_legacy_gate_camera():
    cfg = yaml.safe_load("""
paths: {}
gate:
  camera_id: gate_cam
  read_plates: false
cameras: {}
rules:
  confirm_seconds: 1
  clear_seconds: 1
  min_person_height_px: 1
""")

    cameras = validate_config(cfg)
    assert list(cameras) == ["gate_cam"]
    assert cameras["gate_cam"].capabilities == frozenset({"ppe", "vehicles"})


def test_unknown_capability_is_rejected():
    cfg = {"cameras": {"cam": {"capabilities": ["ppe", "audio"]}}, "rules": {}}

    with pytest.raises(ConfigurationError, match="unsupported values"):
        camera_config(cfg, "cam")


def test_plate_capability_requires_plate_reader():
    cfg = {
        "cameras": {"cam": {"capabilities": ["vehicles", "licence_plates"]}},
        "gate": {"camera_id": "other", "read_plates": False},
        "rules": {},
    }

    with pytest.raises(ConfigurationError, match="read_plates is false"):
        camera_config(cfg, "cam")
