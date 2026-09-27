"""Generate synthetic IoT sensor data for IoTGuard.

`--mode assignment` reproduces the supplied generator semantics.
`--mode month` is the runnable demo mode: it spans ~30 days and interprets the
10% failure rate as device failure-event probability so a 24h target is not
mathematically saturated.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _event_mask(n: int, failure_rate: float, rng: random.Random) -> np.ndarray:
    """Create clustered failure events covering roughly `failure_rate` of samples."""
    mask = np.zeros(n, dtype=bool)
    n_events = max(1, int(round(n * failure_rate / 10)))
    for _ in range(n_events):
        start = rng.randint(0, max(0, n - 16))
        duration = rng.randint(8, 16)
        mask[start : min(n, start + duration)] = True
    return mask


def generate(
    output: str | Path = "data/iot_sensors_stream.csv",
    n_devices: int = 50,
    samples_per_device: int = 2000,
    failure_rate: float = 0.1,
    gap_rate: float = 0.05,
    mode: str = "month",
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    start = datetime(2024, 1, 1)
    data: list[dict] = []

    if mode == "assignment":
        interval_minutes = samples_per_device / (24 * 60 * 30)
    elif mode == "month":
        interval_minutes = (24 * 60 * 30) / samples_per_device
    else:
        raise ValueError("mode must be 'assignment' or 'month'")

    # In month/demo mode, approximately 10% of devices receive clustered failure events.
    failing_devices = set(rng.sample(range(1, n_devices + 1), max(1, round(n_devices * failure_rate)))) if mode == "month" else None

    for device_id in range(1, n_devices + 1):
        event_mask = _event_mask(samples_per_device, failure_rate, rng) if (mode == "month" and device_id in failing_devices) else np.zeros(samples_per_device, dtype=bool)
        for i in range(samples_per_device):
            ts = start + timedelta(minutes=i * interval_minutes)
            base_vib = np.sin(i / 1440 * 2 * np.pi) * 5 + 10
            base_temp = 40 + np.random.normal(0, 2)
            base_press = 100 + np.random.normal(0, 5)
            is_failure_window = bool(event_mask[i]) if mode == "month" else random.random() < failure_rate
            if is_failure_window:
                vib = base_vib + np.random.exponential(10)
                temp = base_temp + np.random.exponential(5)
                press = base_press - np.random.exponential(3)
                status = "fail"
            else:
                vib = base_vib + np.random.normal(0, 2)
                temp = base_temp + np.random.normal(0, 1)
                press = base_press + np.random.normal(0, 2)
                status = "normal" if random.random() < 0.9 else "degraded"

            if random.random() < gap_rate:
                continue

            data.append(
                {
                    "device_id": device_id,
                    "timestamp": ts.isoformat(),
                    "vibration": round(float(vib), 2),
                    "temperature": round(float(temp), 2),
                    "pressure": round(float(press), 2),
                    "status": status,
                }
            )

    df = pd.DataFrame(data)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} readings for {n_devices} devices -> {out}")
    print(f"Mode={mode}; interval_minutes={interval_minutes:.4f}")
    print("failure semantics:", "exact per-reading assignment" if mode == "assignment" else "10% device-event simulation")
    print(df["status"].value_counts(normalize=True).round(4).to_dict())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/iot_sensors_stream.csv")
    parser.add_argument("--devices", type=int, default=50)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--failure-rate", type=float, default=0.10)
    parser.add_argument("--gap-rate", type=float, default=0.05)
    parser.add_argument("--mode", choices=["assignment", "month"], default="month")
    args = parser.parse_args()
    generate(args.output, args.devices, args.samples, args.failure_rate, args.gap_rate, args.mode)
