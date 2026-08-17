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
    build_light_bounds,
    build_stream_message,
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


def baseline_process(frame, bounds, mean_fn):
    return baseline_encode_light_bytes(
        baseline_sample_light_colors(frame, bounds, mean_fn)
    )


def production_mean(region: np.ndarray):
    # OpenCV is the production implementation. NumPy is an exact three-channel
    # compatibility path for this hardware-only benchmark environment.
    return cv2.mean(region) if cv2 is not None else numpy_mean(region)


def payload_checksum(payload) -> int:
    return sum(sum(values) for values in payload.values()) & 0xFFFFFFFF


def assert_equivalent(frame, bounds, mean_fn) -> None:
    baseline_payload = baseline_process(frame, bounds, mean_fn)
    candidate_payload = sample_light_bytes(frame, bounds, mean_fn)
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
        ("write", "HueStream"),
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
    bounds,
    mean_fn,
) -> tuple[float, list[int], float, list[int]]:
    candidate_elapsed: list[float] = []
    baseline_elapsed: list[float] = []
    candidate_checksums: list[int] = []
    baseline_checksums: list[int] = []
    for repeat in range(REPEATS):
        ordered = (
            (("candidate", candidate), ("baseline", baseline))
            if repeat % 2 == 0
            else (("baseline", baseline), ("candidate", candidate))
        )
        for name, processor in ordered:
            checksum = 0
            started = time.perf_counter_ns()
            for frame in frames:
                payload = processor(frame, bounds, mean_fn)
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
    for repeat in range(REPEATS):
        ordered = (
            (("candidate", build_stream_message), ("baseline", baseline_stream_message))
            if repeat % 2 == 0
            else (("baseline", baseline_stream_message), ("candidate", build_stream_message))
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

    candidate_message = build_stream_message(entertainment_id, rgb_bytes)
    baseline_message = baseline_stream_message(entertainment_id, rgb_bytes)
    if candidate_message != baseline_message:
        raise RuntimeError("stream packet differs from baseline implementation")
    if candidate_checksums != baseline_checksums:
        raise RuntimeError("stream packet checksum differs from baseline")
    return (
        statistics.median(candidate_elapsed) / MESSAGE_REPEATS,
        statistics.median(baseline_elapsed) / MESSAGE_REPEATS,
        sum(candidate_message) & 0xFFFFFFFF,
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
    bounds = build_light_bounds(LIGHT_POSITIONS, WIDTH, HEIGHT)
    if any(bottom <= top or right <= left for top, bottom, left, right in bounds.values()):
        raise RuntimeError("benchmark fixture contains an empty light region")

    mean_fn = production_mean
    for frame in frames[:4]:
        assert_equivalent(frame, bounds, mean_fn)

    # Warm up NumPy/OpenCV dispatch and page in the fixture before timing.
    warmup_payload = sample_light_bytes(frames[0], bounds, mean_fn)
    if not warmup_payload:
        raise RuntimeError("benchmark produced an empty payload")

    packet_seconds, packet_baseline_seconds, packet_checksum = measure_stream_pair(
        ENTERTAINMENT_ID,
        warmup_payload,
    )

    (
        candidate_seconds,
        candidate_checksums,
        baseline_seconds,
        baseline_checksums,
    ) = measure_pair(
        sample_light_bytes,
        baseline_process,
        frames,
        bounds,
        mean_fn,
    )

    if len(set(candidate_checksums)) != 1 or len(set(baseline_checksums)) != 1:
        raise RuntimeError(
            f"non-deterministic benchmark output: {candidate_checksums} / {baseline_checksums}"
        )
    if candidate_checksums != baseline_checksums:
        raise RuntimeError("candidate checksum differs from baseline")
    if candidate_checksums[0] != EXPECTED_CHECKSUM:
        raise RuntimeError(
            f"fixture checksum changed: expected {EXPECTED_CHECKSUM}, got {candidate_checksums[0]}"
        )

    latency_us = candidate_seconds * 1_000_000 / FRAME_COUNT
    baseline_latency_us = baseline_seconds * 1_000_000 / FRAME_COUNT
    throughput_fps = FRAME_COUNT / candidate_seconds
    speedup = baseline_seconds / candidate_seconds
    print("EQUIVALENCE baseline=candidate")
    print(f"METRIC stream_packet_build_us={packet_seconds * 1_000_000:.3f}")
    print(f"METRIC stream_packet_baseline_us={packet_baseline_seconds * 1_000_000:.3f}")
    print(
        f"METRIC stream_packet_speedup={packet_baseline_seconds / packet_seconds:.6f}"
    )
    print(f"METRIC stream_flush_before_sleep={flush_before_sleep}")
    print(f"METRIC latest_frames_analyzed={analyzed_frames}")
    print(f"METRIC superseded_frames_dropped={dropped_frames}")
    print(f"METRIC duplicate_frame_generations={duplicate_frames}")
    print(f"METRIC video_latency_us={latency_us:.3f}")
    print(f"METRIC video_throughput_fps={throughput_fps:.3f}")
    print(f"METRIC baseline_latency_us={baseline_latency_us:.3f}")
    print(f"METRIC speedup_vs_baseline={speedup:.6f}")
    print(f"CHECKSUM {candidate_checksums[0]}")
    print(f"PACKET_CHECKSUM {packet_checksum}")


if __name__ == "__main__":
    run()
