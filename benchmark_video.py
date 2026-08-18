#!/usr/bin/env python3
"""Deterministic benchmark for the production frame-analysis path."""

from __future__ import annotations

import statistics
import time
from types import SimpleNamespace
from threading import Event, Thread

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
from video_pipeline import (
    LatestFrameBuffer,
    StreamMessageCache,
    adjust_value_channel,
    build_light_bounds,
    build_stream_message,
    prepare_light_regions,
    sample_bgr_region_bytes,
    sample_light_bytes,
    send_stream_message,
)


WIDTH = 960
HEIGHT = 540
FRAME_COUNT = 64
REPEATS = 5
SEED = 20260817
MESSAGE_REPEATS = 10_000
ENTERTAINMENT_ID = "entertainment-123"
EXPECTED_CHECKSUM = 387910
BRIGHTNESS_VALUE = 30
BRIGHTNESS_FRAME_COUNT = 16
BRIGHTNESS_REPEATS = 5

# Normalized positions model a 16-channel entertainment area around a display.
LIGHT_POSITIONS = {
    "0": (-0.92, 0.0, -0.82),
    "1": (-0.70, 0.0, -0.96),
    "2": (-0.35, 0.0, -0.99),
    "3": (0.00, 0.0, -1.00),
    "4": (0.35, 0.0, -0.99),
    "5": (0.70, 0.0, -0.96),
    "6": (0.92, 0.0, -0.82),
    "7": (0.98, 0.0, -0.25),
    "8": (0.98, 0.0, 0.35),
    "9": (0.92, 0.0, 0.82),
    "10": (0.55, 0.0, 0.98),
    "11": (0.00, 0.0, 1.00),
    "12": (-0.55, 0.0, 0.98),
    "13": (-0.92, 0.0, 0.82),
    "14": (-0.98, 0.0, 0.35),
    "15": (-0.98, 0.0, -0.25),
}


def numpy_mean(region: np.ndarray) -> tuple[float, float, float, float]:
    """Match the three color channels returned by ``cv2.mean``."""
    channels = region.mean(axis=(0, 1))
    return (float(channels[0]), float(channels[1]), float(channels[2]), 0.0)


def baseline_sample_light_colors(frame, bounds, mean_fn):
    """Preserve the pre-refactor per-light sampling implementation."""
    area = {}
    colors = {}
    for light, (top, bottom, left, right) in bounds.items():
        area[light] = frame[top:bottom, left:right, :]
        colors[light] = mean_fn(area[light])
    return colors


def baseline_encode_light_bytes(colors):
    """Preserve the pre-refactor payload encoding implementation."""
    encoded = {}
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


def baseline_stream_message(entertainment_id, rgb_bytes):
    message = bytes("HueStream", "utf-8") + b"\2\0\0\0\0\0\0" + bytes(
        entertainment_id,
        "utf-8",
    )
    for light, payload in rgb_bytes.items():
        message += bytes(chr(int(light)), "utf-8") + payload
    return message


def baseline_adjust_value_channel(hsv, value):
    """Preserve the split/merge brightness implementation."""
    h = hsv[:, :, 0].copy()
    s = hsv[:, :, 1].copy()
    v = hsv[:, :, 2].copy()
    limit = 255 - value
    v[v > limit] = 255
    v[v <= limit] += value
    return np.stack((h, s, v), axis=2)


def candidate_adjust_value_channel(hsv, value):
    return adjust_value_channel(hsv.copy(), value)


def baseline_process(frame, bounds, mean_fn):
    return baseline_encode_light_bytes(
        baseline_sample_light_colors(frame, bounds, mean_fn)
    )


def baseline_bgr_process(frame, bounds, mean_fn):
    rgb_frame = frame[:, :, ::-1].copy()
    return baseline_process(rgb_frame, bounds, mean_fn)


def candidate_bgr_process(frame, regions, mean_fn):
    return sample_bgr_region_bytes(frame, regions, mean_fn)


def production_mean(region: np.ndarray):
    # OpenCV is the production implementation. NumPy is an exact three-channel
    # compatibility path for this hardware-only benchmark environment.
    return cv2.mean(region) if cv2 is not None else numpy_mean(region)


def payload_checksum(payload) -> int:
    return sum(sum(values) for values in payload.values()) & 0xFFFFFFFF


def assert_equivalent(frame, bounds, prepared_bounds, mean_fn) -> None:
    baseline_payload = baseline_process(frame, bounds, mean_fn)
    candidate_payload = sample_light_bytes(frame, prepared_bounds, mean_fn)
    if baseline_payload != candidate_payload:
        raise RuntimeError("candidate payload differs from baseline implementation")


def verify_latest_frame_sync() -> tuple[int, int, int]:
    """Verify stale generations are dropped and never analyzed twice."""
    frame_buffer = LatestFrameBuffer()
    analyzed_generations: list[int] = []
    last_generation = 0

    for generation in (1, 2, 3):
        frame_buffer.publish(np.full((2, 2, 3), generation, dtype=np.uint8))
    result = frame_buffer.next_frame(last_generation)
    if result is None:
        raise RuntimeError("latest-frame buffer did not publish the first frame")
    generation, frame = result
    if int(frame[0, 0, 0]) != generation or generation != 3:
        raise RuntimeError("first analysis did not select the latest generation")
    analyzed_generations.append(generation)
    last_generation = generation

    for generation in (4, 5):
        frame_buffer.publish(np.full((2, 2, 3), generation, dtype=np.uint8))
    result = frame_buffer.next_frame(last_generation)
    if result is None:
        raise RuntimeError("latest-frame buffer did not publish the second frame")
    generation, frame = result
    if int(frame[0, 0, 0]) != generation or generation != 5:
        raise RuntimeError("second analysis did not select the latest generation")
    analyzed_generations.append(generation)
    last_generation = generation

    duplicates = len(analyzed_generations) - len(set(analyzed_generations))
    dropped = 5 - len(analyzed_generations)
    if analyzed_generations != [3, 5] or duplicates != 0 or dropped != 3:
        raise RuntimeError(
            f"unexpected synchronization result: analyzed={analyzed_generations}, "
            f"dropped={dropped}, duplicates={duplicates}"
        )

    frame_buffer.close()
    if frame_buffer.next_frame(last_generation) is not None:
        raise RuntimeError("closed latest-frame buffer returned a frame")

    wait_buffer = LatestFrameBuffer()
    wait_started = Event()
    wait_result = []

    def wait_for_frame():
        wait_started.set()
        wait_result.append(wait_buffer.next_frame(0))

    waiter = Thread(target=wait_for_frame)
    waiter.start()
    if not wait_started.wait(timeout=1):
        raise RuntimeError("latest-frame waiter did not start")
    wait_buffer.publish(np.full((2, 2, 3), 6, dtype=np.uint8))
    waiter.join(timeout=1)
    wait_buffer.close()
    if waiter.is_alive() or len(wait_result) != 1:
        raise RuntimeError("latest-frame waiter was not released by publication")
    if wait_result[0] is None or wait_result[0][0] != 1:
        raise RuntimeError("latest-frame waiter received the wrong generation")
    return len(analyzed_generations), dropped, duplicates


def verify_flush_order() -> int:
    """Verify packets flush before the pacing sleep adds the next-frame delay."""
    events = []

    class RecordingStdin:
        def write(self, text):
            events.append(("write", text))

        def flush(self):
            events.append(("flush",))

    proc = SimpleNamespace(stdin=RecordingStdin())
    send_stream_message(
        proc,
        b"HueStream",
        lambda delay: events.append(("sleep", delay)),
    )
    expected = [
        ("write", b"HueStream"),
        ("flush",),
        ("sleep", 0.0167),
    ]
    if events != expected:
        raise RuntimeError(f"unexpected stream send order: {events}")
    return int(events.index(("flush",)) < events.index(("sleep", 0.0167)))


def measure_pair(
    candidate,
    baseline,
    frames,
    candidate_bounds,
    baseline_bounds,
    mean_fn,
) -> tuple[float, list[int], float, list[int]]:
    candidate_elapsed: list[float] = []
    baseline_elapsed: list[float] = []
    candidate_checksums: list[int] = []
    baseline_checksums: list[int] = []
    for repeat in range(REPEATS):
        ordered = (
            (
                ("candidate", candidate, candidate_bounds),
                ("baseline", baseline, baseline_bounds),
            )
            if repeat % 2 == 0
            else (
                ("baseline", baseline, baseline_bounds),
                ("candidate", candidate, candidate_bounds),
            )
        )
        for name, processor, process_bounds in ordered:
            checksum = 0
            started = time.perf_counter_ns()
            for frame in frames:
                payload = processor(frame, process_bounds, mean_fn)
                checksum = (checksum + payload_checksum(payload)) & 0xFFFFFFFF
            duration = (time.perf_counter_ns() - started) / 1_000_000_000
            if name == "candidate":
                candidate_elapsed.append(duration)
                candidate_checksums.append(checksum)
            else:
                baseline_elapsed.append(duration)
                baseline_checksums.append(checksum)
    return (
        statistics.median(candidate_elapsed),
        candidate_checksums,
        statistics.median(baseline_elapsed),
        baseline_checksums,
    )


def measure_stream_pair(entertainment_id, rgb_bytes):
    candidate_elapsed: list[float] = []
    baseline_elapsed: list[float] = []
    candidate_checksums: list[int] = []
    baseline_checksums: list[int] = []
    candidate_cache = StreamMessageCache(entertainment_id)

    def cached_builder(_entertainment_id, payload):
        return candidate_cache.get(payload)
    for repeat in range(REPEATS):
        ordered = (
            (("candidate", cached_builder), ("baseline", baseline_stream_message))
            if repeat % 2 == 0
            else (("baseline", baseline_stream_message), ("candidate", cached_builder))
        )
        for name, builder in ordered:
            checksum = 0
            started = time.perf_counter_ns()
            for _ in range(MESSAGE_REPEATS):
                message = builder(entertainment_id, rgb_bytes)
                checksum = (checksum + message[0] + len(message)) & 0xFFFFFFFF
            duration = (time.perf_counter_ns() - started) / 1_000_000_000
            if name == "candidate":
                candidate_elapsed.append(duration)
                candidate_checksums.append(checksum)
            else:
                baseline_elapsed.append(duration)
                baseline_checksums.append(checksum)

    candidate_message = candidate_cache.get(rgb_bytes)
    baseline_message = baseline_stream_message(entertainment_id, rgb_bytes)
    if candidate_message != baseline_message:
        raise RuntimeError("stream packet differs from baseline implementation")
    replacement_payload = dict(rgb_bytes)
    replacement_message = candidate_cache.get(replacement_payload)
    replacement_baseline = baseline_stream_message(
        entertainment_id,
        replacement_payload,
    )
    if replacement_message != replacement_baseline:
        raise RuntimeError("stream packet cache failed to refresh")
    candidate_message = candidate_cache.get(rgb_bytes)
    if candidate_message != baseline_message:
        raise RuntimeError("stream packet cache failed to restore")
    if candidate_checksums != baseline_checksums:
        raise RuntimeError("stream packet checksum differs from baseline")
    return (
        statistics.median(candidate_elapsed) / MESSAGE_REPEATS,
        statistics.median(baseline_elapsed) / MESSAGE_REPEATS,
        sum(candidate_message) & 0xFFFFFFFF,
    )


def measure_brightness_pair(frames, value):
    candidate_elapsed: list[float] = []
    baseline_elapsed: list[float] = []
    candidate_checksums: list[int] = []
    baseline_checksums: list[int] = []
    for repeat in range(BRIGHTNESS_REPEATS):
        ordered = (
            (
                ("candidate", candidate_adjust_value_channel),
                ("baseline", baseline_adjust_value_channel),
            )
            if repeat % 2 == 0
            else (
                ("baseline", baseline_adjust_value_channel),
                ("candidate", candidate_adjust_value_channel),
            )
        )
        for name, adjuster in ordered:
            checksum = 0
            started = time.perf_counter_ns()
            for frame in frames:
                output = adjuster(frame, value)
                checksum = (checksum + int(output[0, 0, 2])) & 0xFFFFFFFF
            duration = (time.perf_counter_ns() - started) / 1_000_000_000
            if name == "candidate":
                candidate_elapsed.append(duration)
                candidate_checksums.append(checksum)
            else:
                baseline_elapsed.append(duration)
                baseline_checksums.append(checksum)
    if candidate_checksums != baseline_checksums:
        raise RuntimeError("brightness output checksum differs from baseline")
    return (
        statistics.median(candidate_elapsed) / len(frames),
        statistics.median(baseline_elapsed) / len(frames),
    )


def run() -> None:
    analyzed_frames, dropped_frames, duplicate_frames = verify_latest_frame_sync()
    flush_before_sleep = verify_flush_order()
    rng = np.random.default_rng(SEED)
    frames = rng.integers(
        0,
        256,
        size=(FRAME_COUNT, HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )
    brightness_frames = rng.integers(
        0,
        256,
        size=(BRIGHTNESS_FRAME_COUNT, HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )
    for frame in brightness_frames[:4]:
        baseline_output = baseline_adjust_value_channel(frame, BRIGHTNESS_VALUE)
        candidate_output = candidate_adjust_value_channel(frame, BRIGHTNESS_VALUE)
        if not np.array_equal(baseline_output, candidate_output):
            raise RuntimeError("brightness output differs from baseline implementation")
    candidate_adjust_value_channel(brightness_frames[0], BRIGHTNESS_VALUE)
    brightness_seconds, brightness_baseline_seconds = measure_brightness_pair(
        brightness_frames,
        BRIGHTNESS_VALUE,
    )
    for value in (0, 1, BRIGHTNESS_VALUE, 255):
        for frame in brightness_frames[:2]:
            baseline_output = baseline_adjust_value_channel(frame, value)
            candidate_output = candidate_adjust_value_channel(frame, value)
            if not np.array_equal(baseline_output, candidate_output):
                raise RuntimeError(
                    f"brightness output differs for value {value}"
                )
    bounds = build_light_bounds(LIGHT_POSITIONS, WIDTH, HEIGHT)
    if any(bottom <= top or right <= left for top, bottom, left, right in bounds.values()):
        raise RuntimeError("benchmark fixture contains an empty light region")
    prepared_bounds = tuple(bounds.items())
    prepared_regions = prepare_light_regions(prepared_bounds)

    mean_fn = production_mean
    for frame in frames[:4]:
        assert_equivalent(frame, bounds, prepared_bounds, mean_fn)
    for frame in frames[:4]:
        baseline_bgr = baseline_bgr_process(frame, bounds, mean_fn)
        candidate_bgr = candidate_bgr_process(frame, prepared_regions, mean_fn)
        if baseline_bgr != candidate_bgr:
            raise RuntimeError("BGR sampling output differs from baseline")

    # Warm up NumPy/OpenCV dispatch and page in the fixture before timing.
    warmup_payload = sample_light_bytes(frames[0], prepared_bounds, mean_fn)
    if not warmup_payload:
        raise RuntimeError("benchmark produced an empty payload")

    packet_seconds, packet_baseline_seconds, packet_checksum = measure_stream_pair(
        ENTERTAINMENT_ID,
        warmup_payload,
    )

    (
        rgb_seconds,
        rgb_candidate_checksums,
        rgb_baseline_seconds,
        rgb_baseline_checksums,
    ) = measure_pair(
        sample_light_bytes,
        baseline_process,
        frames,
        prepared_bounds,
        bounds,
        mean_fn,
    )
    (
        bgr_seconds,
        bgr_candidate_checksums,
        bgr_baseline_seconds,
        bgr_baseline_checksums,
    ) = measure_pair(
        candidate_bgr_process,
        baseline_bgr_process,
        frames,
        prepared_regions,
        bounds,
        mean_fn,
    )

    if len(set(rgb_candidate_checksums)) != 1 or len(set(rgb_baseline_checksums)) != 1:
        raise RuntimeError(
            f"non-deterministic RGB output: {rgb_candidate_checksums} / {rgb_baseline_checksums}"
        )
    if rgb_candidate_checksums != rgb_baseline_checksums:
        raise RuntimeError("RGB candidate checksum differs from baseline")
    if len(set(bgr_candidate_checksums)) != 1 or len(set(bgr_baseline_checksums)) != 1:
        raise RuntimeError(
            f"non-deterministic BGR output: {bgr_candidate_checksums} / {bgr_baseline_checksums}"
        )
    if bgr_candidate_checksums != bgr_baseline_checksums:
        raise RuntimeError("BGR candidate checksum differs from baseline")
    if rgb_candidate_checksums[0] != EXPECTED_CHECKSUM:
        raise RuntimeError(
            f"fixture checksum changed: expected {EXPECTED_CHECKSUM}, got {rgb_candidate_checksums[0]}"
        )
    if bgr_candidate_checksums[0] != EXPECTED_CHECKSUM:
        raise RuntimeError(
            f"BGR fixture checksum changed: expected {EXPECTED_CHECKSUM}, got {bgr_candidate_checksums[0]}"
        )

    latency_us = bgr_seconds * 1_000_000 / FRAME_COUNT
    baseline_latency_us = bgr_baseline_seconds * 1_000_000 / FRAME_COUNT
    throughput_fps = FRAME_COUNT / bgr_seconds
    speedup = bgr_baseline_seconds / bgr_seconds
    rgb_latency_us = rgb_seconds * 1_000_000 / FRAME_COUNT
    rgb_baseline_latency_us = rgb_baseline_seconds * 1_000_000 / FRAME_COUNT
    rgb_speedup = rgb_baseline_seconds / rgb_seconds
    bgr_latency_us = latency_us
    bgr_baseline_latency_us = baseline_latency_us
    bgr_speedup = speedup
    print("EQUIVALENCE baseline=candidate")
    print(f"METRIC brightness_adjust_us={brightness_seconds * 1_000_000:.3f}")
    print(
        f"METRIC brightness_baseline_us={brightness_baseline_seconds * 1_000_000:.3f}"
    )
    print(
        f"METRIC brightness_speedup={brightness_baseline_seconds / brightness_seconds:.6f}"
    )
    print(f"METRIC stream_packet_build_us={packet_seconds * 1_000_000:.3f}")
    print(f"METRIC stream_packet_baseline_us={packet_baseline_seconds * 1_000_000:.3f}")
    print(
        f"METRIC stream_packet_speedup={packet_baseline_seconds / packet_seconds:.6f}"
    )
    print(f"METRIC bgr_latency_us={bgr_latency_us:.3f}")
    print(f"METRIC bgr_baseline_latency_us={bgr_baseline_latency_us:.3f}")
    print(f"METRIC bgr_speedup={bgr_speedup:.6f}")
    print(f"METRIC rgb_latency_us={rgb_latency_us:.3f}")
    print(f"METRIC rgb_baseline_latency_us={rgb_baseline_latency_us:.3f}")
    print(f"METRIC rgb_speedup={rgb_speedup:.6f}")
    print(f"METRIC stream_flush_before_sleep={flush_before_sleep}")
    print(f"METRIC latest_frames_analyzed={analyzed_frames}")
    print(f"METRIC superseded_frames_dropped={dropped_frames}")
    print(f"METRIC duplicate_frame_generations={duplicate_frames}")
    print(f"METRIC video_latency_us={latency_us:.3f}")
    print(f"METRIC video_throughput_fps={throughput_fps:.3f}")
    print(f"METRIC baseline_latency_us={baseline_latency_us:.3f}")
    print(f"METRIC speedup_vs_baseline={speedup:.6f}")
    print(f"CHECKSUM {bgr_candidate_checksums[0]}")
    print(f"PACKET_CHECKSUM {packet_checksum}")


if __name__ == "__main__":
    run()
