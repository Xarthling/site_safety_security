from safety.configuration import CameraConfig, CameraSource, apply_camera_config, camera_config


def test_persisted_camera_overlay_wins_over_yaml_camera_settings():
    cfg = {
        "detector": {"weights": "yaml.pt"},
        "cameras": {
            "cam": {
                "name": "YAML camera",
                "enabled": True,
                "capabilities": ["ppe"],
                "zones": {"detection_zone": [[0, 0], [1, 0], [1, 1]]},
                "rules": {"confirm_seconds": 99},
            }
        },
    }
    persisted = CameraConfig(
        id="cam",
        name="Dashboard camera",
        enabled=False,
        source=CameraSource("recorded", "data/clips/updated.mp4"),
        capabilities=frozenset({"ppe"}),
        zones={"detection_zone": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]},
        rules={"confirm_seconds": 3},
        vehicle=None,
    )

    effective = camera_config(apply_camera_config(cfg, persisted), "cam")

    assert effective == persisted


def test_persisted_vehicle_settings_override_legacy_gate_yaml():
    cfg = {
        "gate": {"camera_id": "gate_cam", "conf": 0.35},
        "cameras": {
            "gate_cam": {
                "capabilities": ["vehicles"],
                "vehicle": {"conf": 0.72},
            }
        },
    }

    assert camera_config(cfg, "gate_cam").vehicle["conf"] == 0.72
