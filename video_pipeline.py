"""Frame-analysis primitives shared by the runtime and benchmark.

The runtime supplies ``cv2.mean`` as the mean function.  Keeping the sampling
loop here makes the measured workload the same code path used by production,
while the benchmark can run without a camera, bridge, or OpenCV installation.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from threading import Condition
import time

import numpy as np


MeanFunction = Callable[[np.ndarray], Sequence[float]]
Bounds = Mapping[str, tuple[int, int, int, int]]
Bound = tuple[str, tuple[int, int, int, int]]
PreparedBounds = tuple[Bound, ...]
PreparedRegion = tuple[str, slice, slice]
PreparedRegions = tuple[PreparedRegion, ...]
RGB_CHANNEL_ORDER = (0, 1, 2)


@lru_cache(maxsize=256)
def _brightness_lut(value: int) -> np.ndarray:
    return np.minimum(
        np.arange(256, dtype=np.uint16) + value,
        255,
    ).astype(np.uint8)


def adjust_value_channel(hsv: np.ndarray, value: int) -> np.ndarray:
    """Apply the production saturating brightness shift in place."""
    value_channel = hsv[:, :, 2]
    np.take(_brightness_lut(value), value_channel, out=value_channel)
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
            self._condition.notify_all()
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


def prepare_light_regions(bounds: PreparedBounds) -> PreparedRegions:
    """Precompute row and column slices for repeated frame analysis."""
    return tuple(
        (light, slice(top, bottom), slice(left, right))
        for light, (top, bottom, left, right) in bounds
    )


def sample_light_bytes(
    frame: np.ndarray,
    bounds: Bounds | PreparedBounds,
    mean_fn: MeanFunction,
    channel_order: tuple[int, int, int] = RGB_CHANNEL_ORDER,
) -> dict[str, bytes]:
    """Average regions and encode Hue bytes in the requested channel order."""
    bound_items = bounds.items() if isinstance(bounds, Mapping) else bounds
    encoded: dict[str, bytes] = {}
    for light, (top, bottom, left, right) in bound_items:
        color = mean_fn(frame[top:bottom, left:right, :])
        red = int(color[channel_order[0]] / 2)
        green = int(color[channel_order[1]] / 2)
        blue = int(color[channel_order[2]] / 2)
        encoded[light] = bytes((red, red, green, green, blue, blue))
    return encoded


def sample_bgr_region_bytes(
    frame: np.ndarray,
    regions: PreparedRegions,
    mean_fn: MeanFunction,
) -> dict[str, bytes]:
    """Average pre-sliced BGR regions and encode their Hue payload as RGB."""
    encoded: dict[str, bytes] = {}
    for light, row_slice, column_slice in regions:
        color = mean_fn(frame[row_slice, column_slice])
        red = int(color[2]) >> 1
        green = int(color[1]) >> 1
        blue = int(color[0]) >> 1
        encoded[light] = bytes((red, red, green, green, blue, blue))
    return encoded


_STREAM_HEADER = b"HueStream" + b"\2\0\0\0\0\0\0"


def _build_stream_message(
    header: bytes,
    rgb_bytes: Mapping[str, bytes],
    light_ids: Sequence[int] | None = None,
    message: bytearray | None = None,
) -> bytes:
    if message is None:
        message = bytearray(header)
    else:
        del message[len(header):]
    append = message.append
    extend = message.extend
    if light_ids is None:
        for light, payload in rgb_bytes.items():
            append(int(light))
            extend(payload)
    else:
        for light_id, payload in zip(light_ids, rgb_bytes.values()):
            append(light_id)
            extend(payload)
    return bytes(message)


def build_stream_message(
    entertainment_id: str,
    rgb_bytes: Mapping[str, bytes],
) -> bytes:
    """Assemble a Hue packet without repeated immutable-byte concatenation."""
    return _build_stream_message(
        _STREAM_HEADER + entertainment_id.encode("utf-8"),
        rgb_bytes,
    )


class StreamMessageCache:
    """Reuse a packet while the analyzer payload mapping is unchanged."""

    def __init__(self, entertainment_id: str) -> None:
        self._header = _STREAM_HEADER + entertainment_id.encode("utf-8")
        self._payload: Mapping[str, bytes] | None = None
        self._light_keys: tuple[str, ...] | None = None
        self._light_ids: tuple[int, ...] = ()
        self._message_buffer = bytearray(self._header)
        self._message = b""

    def get(self, rgb_bytes: Mapping[str, bytes]) -> bytes:
        if rgb_bytes is self._payload:
            return self._message
        light_keys = tuple(rgb_bytes)
        if light_keys != self._light_keys:
            self._light_keys = light_keys
            self._light_ids = tuple(map(int, light_keys))
        self._message = _build_stream_message(
            self._header,
            rgb_bytes,
            self._light_ids,
            self._message_buffer,
        )
        self._payload = rgb_bytes
        return self._message


def send_stream_message(proc, message: bytes, sleep_fn: Callable[[float], None]) -> None:
    """Flush each binary Hue packet before pacing the next packet."""
    proc.stdin.write(message)
    proc.stdin.flush()
    sleep_fn(0.0167)


def send_stream_message_on_schedule(
    proc,
    message_supplier: Callable[[], bytes],
    sleep_fn: Callable[[float], None],
    next_deadline: float,
    interval: float = 0.0167,
    monotonic_fn: Callable[[], float] = time.monotonic,
    flush: bool = True,
) -> float:
    """Pace before each send, resolving the supplier immediately before writing."""
    now = monotonic_fn()
    delay = next_deadline - now
    if delay > 0:
        sleep_fn(delay)
    else:
        next_deadline = now
    message = message_supplier()
    stdin = proc.stdin
    stdin.write(message)
    if flush:
        stdin.flush()
    return next_deadline + interval




