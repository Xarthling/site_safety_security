import pytest

from safety.configuration import ConfigurationError, validate_zone_points


def test_normalized_zone_points_are_returned_as_numeric_tuples():
    assert validate_zone_points([[0, 0], ["0.5", "1"], [1, 0]]) == (
        (0.0, 0.0), (0.5, 1.0), (1.0, 0.0)
    )


@pytest.mark.parametrize("points, message", [
    ([[0, 0], [1, 1]], "at least 3"),
    ([[0, 0], [1, 1], [2, 0]], "between 0 and 1"),
    ([[0, 0], ["x", 1], [1, 0]], "must contain numbers"),
    ([[0, 0, 1], [1, 1], [1, 0]], "exactly x and y"),
])
def test_invalid_zone_points_are_rejected(points, message):
    with pytest.raises(ConfigurationError, match=message):
        validate_zone_points(points)
