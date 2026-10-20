import pytest

from safety.ingest import probe, read_frames
from tests.helpers import make_video


def test_probe_reports_clip_properties(tmp_path):
    path = str(tmp_path / "c.mp4")
    make_video(path, fps=10, seconds=3)
    info = probe(path, target_fps=5)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10)
    assert info.stride == 2 and info.sample_fps == pytest.approx(5)
    assert info.duration_s == pytest.approx(3, abs=0.2)


def test_frames_are_sampled_and_stamped_in_video_time(tmp_path):
    path = str(tmp_path / "c.mp4")
    make_video(path, fps=10, seconds=3)
    frames = list(read_frames(path, target_fps=5))
    assert len(frames) == 15
    assert [round(f.video_time_s, 2) for f in frames[:3]] == [0.0, 0.2, 0.4]
    assert frames[1].index == 2


def test_max_frames_limits_output(tmp_path):
    path = str(tmp_path / "c.mp4")
    make_video(path)
    assert len(list(read_frames(path, target_fps=5, max_frames=4))) == 4


def test_target_above_file_rate_keeps_every_frame(tmp_path):
    path = str(tmp_path / "c.mp4")
    make_video(path, fps=10, seconds=1)
    assert len(list(read_frames(path, target_fps=30))) == 10


def test_missing_and_invalid_files_raise_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(str(tmp_path / "nope.mp4"))
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(ValueError):
        probe(str(bad))
