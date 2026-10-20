import numpy as np

from live import LiveWall, fit, make_grid
from run import load_config
from safety.storage import connect, list_events, start_run
from tests.helpers import SixPeopleDetector, make_video


def setup(tmp_path, n_cameras=2, seconds=8):
    cfg = load_config("config.yaml")
    cfg["paths"]["snapshots"] = str(tmp_path / "snaps")
    jobs = []
    for i in range(n_cameras):
        path = str(tmp_path / f"cam{i}.mp4")
        make_video(path, fps=10, seconds=seconds, size=(640, 480))
        jobs.append((path, f"cam_{i}"))
    return cfg, jobs


def play(wall, seconds, dt=0.1):
    now = 0.0
    while now <= seconds:
        grid = wall.step(now)
        now += dt
    return grid


def test_fit_keeps_shape_and_pads():
    out = fit(np.full((480, 640, 3), 200, dtype=np.uint8), 640, 360)
    assert out.shape == (360, 640, 3)
    assert out[:, :40].sum() == 0 and out[:, 300].sum() > 0  # black bars left, picture in the middle


def test_grid_layout_pads_the_last_row():
    tiles = [np.full((10, 20, 3), i + 1, dtype=np.uint8) for i in range(3)]
    grid = make_grid(tiles, cols=2)
    assert grid.shape == (20, 40, 3)
    assert grid[15, 30].sum() == 0  # the empty fourth slot is black


def test_two_cameras_play_in_real_time_and_save_events(tmp_path):
    cfg, jobs = setup(tmp_path)
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "m", "c", [])
    wall = LiveWall(jobs, SixPeopleDetector(), cfg, conn, run_id, tile_size=(320, 180))
    grid = play(wall, 9.0)
    assert grid.shape == (180, 640, 3)  # two 320x180 tiles side by side
    assert wall.all_done
    rows = list_events(conn, run_id=run_id)
    assert len(rows) == 10 and {r["camera_id"] for r in rows} == {"cam_0", "cam_1"}
    assert all(c.n_events == 5 for c in wall.cameras)
    wall.close()


def test_alert_banner_shows_then_expires(tmp_path):
    cfg, jobs = setup(tmp_path, n_cameras=1)
    wall = LiveWall(jobs, SixPeopleDetector(), cfg, tile_size=(320, 180))
    cam = wall.cameras[0]
    play(wall, 3.5)
    assert cam.banner and cam.banner_until > 3.5       # first events confirm at about 3 s
    banner_row = wall.step(3.6)[180 - 20, 5]            # bottom bar of the tile
    assert banner_row[2] > banner_row[0]                # red-ish
    wall.step(cam.banner_until + 0.5)
    assert wall.step(cam.banner_until + 0.6)[180 - 20, 5].sum() < 200 * 3  # bar gone (clip frame is grey)


def test_falling_behind_drops_frames_instead_of_stalling(tmp_path):
    cfg, jobs = setup(tmp_path, n_cameras=1)
    detector = SixPeopleDetector()
    wall = LiveWall(jobs, detector, cfg, tile_size=(320, 180))
    wall.step(0.0)
    wall.step(5.0)                                      # the model "took 5 seconds": jump ahead
    cam = wall.cameras[0]
    assert detector.calls == 2 and cam.video_t >= 5.0 - 0.11
    assert wall.summary(5.0)                            # summary lines are produced


def test_watch_only_mode_saves_nothing(tmp_path):
    cfg, jobs = setup(tmp_path, n_cameras=1)
    wall = LiveWall(jobs, SixPeopleDetector(), cfg, conn=None, tile_size=(320, 180))
    play(wall, 9.0)
    assert wall.all_done and wall.cameras[0].n_events == 5


def test_gate_camera_logs_vehicles_in_the_live_wall(tmp_path):
    from run import VehicleModels
    from safety.plates import PlateRead
    from tests.helpers import CarDetector, FakePlateReader
    cfg, jobs = setup(tmp_path, n_cameras=1)
    jobs = [(jobs[0][0], "gate_cam")]
    models = VehicleModels(CarDetector(first=2, last=12),
                           FakePlateReader([[PlateRead("ABC123", 0.9, (30.0, 20.0, 120.0, 60.0))]]))
    conn = connect(str(tmp_path / "e.db"))
    run_id = start_run(conn, "m", "c", [])
    wall = LiveWall(jobs, SixPeopleDetector(), cfg, conn, run_id, tile_size=(320, 180), vehicle_models=models)
    cam = wall.cameras[0]
    seen_vehicle_banner = False
    now = 0.0
    while now <= 9.0:
        wall.step(now)
        seen_vehicle_banner = seen_vehicle_banner or cam.banner.startswith("VEHICLE")
        now += 0.1
    assert cam.n_vehicles == 1 and seen_vehicle_banner and "ABC123" in cam.banner
    from safety.storage import list_vehicles
    assert [r["plate"] for r in list_vehicles(conn, run_id=run_id)] == ["ABC123"]
    assert "vehicles" in wall.summary(9.0)[0]
