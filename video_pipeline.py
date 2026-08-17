"""Frame-analysis primitives shared by the runtime and benchmark.

The runtime supplies ``cv2.mean`` as the mean function.  Keeping the sampling
loop here makes the measured workload the same code path used by production,
while the benchmark can run without a camera, bridge, or OpenCV installation.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from threading import Condition

import numpy as np


MeanFunction = Callable[[np.ndarray], Sequence[float]]
Bounds = Mapping[str, tuple[int, int, int, int]]


@lru_cache(maxsize=256)
def _brightness_lut(value: int) -> np.ndarray:
    return np.minimum(
        np.arange(256, dtype=np.uint16) + value,
        255,
    ).astype(np.uint8)


def adjust_value_channel(hsv: np.ndarray, value: int) -> np.ndarray:
    """Apply the production saturating brightness shift in place."""
    hsv[:, :, 2] = _brightness_lut(value)[hsv[:, :, 2]]
    return hsv


class LatestFrameBuffer:
    """Publish frames while allowing consumers to skip superseded generations."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._frame: np.ndarray | None = None
        self._generation = 0
        self._closed = False

    def publish(self, frame: np.ndarray) -> int:
        with self._condition:
            self._frame = frame
            self._generation += 1
            generation = self._generation
            self._condition.notify()
            return generation

    def next_frame(self, last_generation: int) -> tuple[int, np.ndarray] | None:
        with self._condition:
            while not self._closed and self._generation <= last_generation:
                self._condition.wait()
            if self._closed:
                return None
            return self._generation, self._frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


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


def sample_light_bytes(
    frame: np.ndarray,
    bounds: Bounds,
    mean_fn: MeanFunction,
) -> dict[str, bytearray]:
    """Average each light region and encode its Hue payload in one pass."""
    encoded: dict[str, bytearray] = {}
    for light, (top, bottom, left, right) in bounds.items():
        color = mean_fn(frame[top:bottom, left:right, :])
        red = int(color[0] / 2)
        green = int(color[1] / 2)
        blue = int(color[2] / 2)
        encoded[light] = bytearray((red, red, green, green, blue, blue))
    return encoded


def build_stream_message(
    entertainment_id: str,
    rgb_bytes: Mapping[str, bytearray],
) -> bytes:
    """Assemble a Hue packet without repeated immutable-byte concatenation."""
    message = bytearray(
        b"HueStream"
        + b"\2\0\0\0\0\0\0"
        + entertainment_id.encode("utf-8")
    )
    for light, payload in rgb_bytes.items():
        message.append(int(light))
        message.extend(payload)
    return bytes(message)


def send_stream_message(proc, message: bytes, sleep_fn: Callable[[float], None]) -> None:
    """Flush each binary Hue packet before pacing the next packet."""
    proc.stdin.write(message)
    proc.stdin.flush()
    sleep_fn(0.0167)




