"""Frame-analysis primitives shared by the runtime and benchmark.

The runtime supplies ``cv2.mean`` as the mean function.  Keeping the sampling
loop here makes the measured workload the same code path used by production,
while the benchmark can run without a camera, bridge, or OpenCV installation.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np


MeanFunction = Callable[[np.ndarray], Sequence[float]]
Bounds = Mapping[str, tuple[int, int, int, int]]


def build_light_bounds(
    light_positions: Mapping[str, Sequence[float]],
    width: int,
    height: int,
    breadth: float = 0.15,
) -> dict[str, tuple[int, int, int, int]]:
    """Convert normalized light positions to image slices.

    Bounds use the production order ``top, bottom, left, right``.  The
    conversion intentionally matches ``harmonize.py``: only lower bounds are
    clamped, while OpenCV/NumPy naturally clip upper bounds at the frame edge.
    """
    distance = int(breadth * (width / 2 + height / 2))
    bounds: dict[str, tuple[int, int, int, int]] = {}
    for light, position in light_positions.items():
        x = (position[0] + 1) * width // 2
        y = (-position[2] + 1) * height // 2
        top, bottom = y - distance, y + distance
        left, right = x - distance, x + distance
        bounds[light] = tuple(
            max(0, int(value)) for value in (top, bottom, left, right)
        )
    return bounds


def sample_light_colors(
    frame: np.ndarray,
    bounds: Bounds,
    mean_fn: MeanFunction,
) -> dict[str, Sequence[float]]:
    """Average each configured light region from one RGB frame."""
    colors: dict[str, Sequence[float]] = {}
    for light, (top, bottom, left, right) in bounds.items():
        colors[light] = mean_fn(frame[top:bottom, left:right, :])
    return colors


def encode_light_bytes(
    colors: Mapping[str, Sequence[float]],
) -> dict[str, bytearray]:
    """Convert averaged RGB values to the Hue streaming payload format."""
    encoded: dict[str, bytearray] = {}
    for light, color in colors.items():
        encoded[light] = bytearray(
            (
                int(color[0] / 2),
                int(color[0] / 2),
                int(color[1] / 2),
                int(color[1] / 2),
                int(color[2] / 2),
                int(color[2] / 2),
            )
        )
    return encoded
