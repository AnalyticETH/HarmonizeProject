#!/usr/bin/env python3
"""Deterministic benchmark for the production frame-analysis path."""

from __future__ import annotations

import statistics
import time

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from video_pipeline import (
    build_light_bounds,
    encode_light_bytes,
    sample_light_colors,
)


WIDTH = 960
HEIGHT = 540
FRAME_COUNT = 64
REPEATS = 5
SEED = 20260817
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


def production_mean(region: np.ndarray):
    # OpenCV is the production implementation. NumPy is an exact three-channel
    # compatibility path for this hardware-only benchmark environment.
    return cv2.mean(region) if cv2 is not None else numpy_mean(region)


def payload_checksum(payload) -> int:
    return sum(sum(values) for values in payload.values()) & 0xFFFFFFFF


def assert_equivalent(frame, bounds, mean_fn) -> None:
    baseline_colors = baseline_sample_light_colors(frame, bounds, mean_fn)
    candidate_colors = sample_light_colors(frame, bounds, mean_fn)
    for light in bounds:
        if not np.allclose(
            np.asarray(baseline_colors[light]),
            np.asarray(candidate_colors[light]),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(f"sampling output mismatch for light {light}")

    baseline_payload = baseline_encode_light_bytes(baseline_colors)
    candidate_payload = encode_light_bytes(candidate_colors)
    if baseline_payload != candidate_payload:
        raise RuntimeError("encoded payload differs from baseline implementation")


def measure(analyzer, encoder, frames, bounds, mean_fn) -> tuple[float, list[int]]:
    elapsed: list[float] = []
    checksums: list[int] = []
    for _ in range(REPEATS):
        checksum = 0
        started = time.perf_counter_ns()
        for frame in frames:
            colors = analyzer(frame, bounds, mean_fn)
            payload = encoder(colors)
            checksum = (checksum + payload_checksum(payload)) & 0xFFFFFFFF
        elapsed.append((time.perf_counter_ns() - started) / 1_000_000_000)
        checksums.append(checksum)
    return statistics.median(elapsed), checksums


def run() -> None:
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
    warmup_colors = sample_light_colors(frames[0], bounds, mean_fn)
    warmup_payload = encode_light_bytes(warmup_colors)
    if not warmup_payload:
        raise RuntimeError("benchmark produced an empty payload")

    candidate_seconds, candidate_checksums = measure(
        sample_light_colors,
        encode_light_bytes,
        frames,
        bounds,
        mean_fn,
    )
    baseline_seconds, baseline_checksums = measure(
        baseline_sample_light_colors,
        baseline_encode_light_bytes,
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
    print(f"METRIC video_latency_us={latency_us:.3f}")
    print(f"METRIC video_throughput_fps={throughput_fps:.3f}")
    print(f"METRIC baseline_latency_us={baseline_latency_us:.3f}")
    print(f"METRIC speedup_vs_baseline={speedup:.6f}")
    print(f"CHECKSUM {candidate_checksums[0]}")


if __name__ == "__main__":
    run()
