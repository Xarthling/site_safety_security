import numpy as np
import pytest

from safety.plates import PlateReader, normalise_plate, ocr_confidence, plate_for_storage


def test_normalise_keeps_letters_and_digits_only():
    assert normalise_plate("abc-123 ") == "ABC123"
    assert normalise_plate("1ab 2cd") == "1AB2CD"


def test_ocr_confidence_handles_one_value_or_padded_list():
    assert ocr_confidence(0.9, 6) == pytest.approx(0.9)
    assert ocr_confidence([1.0, 0.8, 0.9, 0.9, 0.0, 0.0, 0.0], 4) == pytest.approx(0.9)  # padding ignored
    assert ocr_confidence([], 3) == 0.0


def test_plate_storage_modes():
    assert plate_for_storage("ABC123", "text") == "ABC123"
    assert plate_for_storage("ABC123", "none") is None
    assert plate_for_storage(None, "text") is None
    hashed = plate_for_storage("ABC123", "hash", "salt1")
    assert hashed != "ABC123" and len(hashed) == 12
    assert hashed == plate_for_storage("ABC123", "hash", "salt1")      # same plate, same fingerprint
    assert hashed != plate_for_storage("ABC123", "hash", "salt2")      # the salt changes it
    assert hashed != plate_for_storage("ABD123", "hash", "salt1")
    with pytest.raises(ValueError):
        plate_for_storage("ABC123", "everything")


def test_real_plate_reader_on_a_rendered_plate():
    """Runs the actual fast-alpr models (they download on first use); skipped if unavailable."""
    pytest.importorskip("fast_alpr")
    import cv2
    try:
        reader = PlateReader(ocr_device="cpu")
    except Exception as e:  # for example no internet to fetch the models
        pytest.skip(f"plate models could not be loaded: {e}")
    plate = np.full((80, 240, 3), 255, np.uint8)
    cv2.rectangle(plate, (2, 2), (237, 77), (0, 0, 0), 3)
    cv2.putText(plate, "ABC123", (18, 58), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 3)
    scene = np.full((480, 640, 3), 110, np.uint8)
    scene[200:280, 200:440] = plate
    reads = reader(scene)
    assert [r.text for r in reads] == ["ABC123"] and reads[0].conf > 0.8
    assert reader(np.full((480, 640, 3), 90, np.uint8)) == []
