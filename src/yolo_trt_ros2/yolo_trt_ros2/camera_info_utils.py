"""Helpers for constructing strictly sized sensor_msgs/CameraInfo arrays."""

from array import array


def _matrix_array(field_name, rows, expected_rows, expected_cols):
    if len(rows) != expected_rows:
        raise ValueError(
            '%s must have %d rows, got %d'
            % (field_name, expected_rows, len(rows))
        )
    for index, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(
                '%s row %d must have %d columns, got %d'
                % (field_name, index, expected_cols, len(row))
            )
    values = array('d', (float(value) for row in rows for value in row))
    expected_size = expected_rows * expected_cols
    if len(values) != expected_size:
        raise ValueError(
            '%s must contain %d values, got %d'
            % (field_name, expected_size, len(values))
        )
    return values


def set_camera_info_intrinsics(msg, fx, fy, ppx, ppy, coeffs):
    """Populate CameraInfo D/K/R/P with explicit matrix dimensions."""
    msg.d = array('d', (float(value) for value in coeffs))
    msg.k = _matrix_array(
        'CameraInfo.k',
        (
            (fx, 0.0, ppx),
            (0.0, fy, ppy),
            (0.0, 0.0, 1.0),
        ),
        3,
        3,
    )
    msg.r = _matrix_array(
        'CameraInfo.r',
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        3,
        3,
    )
    msg.p = _matrix_array(
        'CameraInfo.p',
        (
            (fx, 0.0, ppx, 0.0),
            (0.0, fy, ppy, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        3,
        4,
    )
