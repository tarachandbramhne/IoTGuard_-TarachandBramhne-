"""IoTGuard preprocessing: hourly resampling, gap repair, scaling, sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["vibration", "temperature", "pressure"]
LOOKBACK_HOURS = 24
HORIZON_HOURS = 24


def hourly_resample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["device_id", "timestamp"]).sort_values(["device_id", "timestamp"])
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    frames: list[pd.DataFrame] = []
    for device_id, g in df.groupby("device_id"):
        g = g.set_index("timestamp")[FEATURES + ["status"]]
        hourly = g[FEATURES].resample("1h").mean()
        failure_flag = g["status"].eq("fail").resample("1h").max().fillna(False).astype(int)
        degraded_flag = g["status"].eq("degraded").resample("1h").max().fillna(False).astype(int)
        hourly["failure_flag"] = failure_flag
        hourly["degraded_flag"] = degraded_flag
        hourly["device_id"] = int(device_id)
        frames.append(hourly.reset_index())
    return pd.concat(frames, ignore_index=True)


def repair_and_scale(hourly: pd.DataFrame, scaler_path: str | Path | None = None):
    result: list[pd.DataFrame] = []
    scalers: dict[str, StandardScaler] = {}
    for device_id, g in hourly.groupby("device_id"):
        g = g.sort_values("timestamp").copy()
        # First repair short gaps by forward fill, then KNN for any remaining holes.
        g[FEATURES] = g[FEATURES].ffill().bfill()
        imputer = KNNImputer(n_neighbors=3)
        g[FEATURES] = imputer.fit_transform(g[FEATURES])
        scaler = StandardScaler()
        g[FEATURES] = scaler.fit_transform(g[FEATURES])
        scalers[str(int(device_id))] = scaler
        result.append(g)
    repaired = pd.concat(result, ignore_index=True)
    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scalers, scaler_path)
    return repaired, scalers


def make_sequences(repaired: pd.DataFrame):
    X, y, ends, devices = [], [], [], []
    for device_id, g in repaired.groupby("device_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        values = g[FEATURES].to_numpy(dtype=np.float32)
        fails = g["failure_flag"].to_numpy(dtype=np.int8)
        max_start = len(g) - LOOKBACK_HOURS - HORIZON_HOURS + 1
        for i in range(max(0, max_start)):
            X.append(values[i : i + LOOKBACK_HOURS])
            future = fails[i + LOOKBACK_HOURS : i + LOOKBACK_HOURS + HORIZON_HOURS]
            y.append(float(future.max() if len(future) else 0))
            ends.append(str(g.loc[i + LOOKBACK_HOURS - 1, "timestamp"]))
            devices.append(int(device_id))
    if not X:
        raise RuntimeError("No sequences were created. Check cadence and input data.")
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(ends),
        np.asarray(devices, dtype=np.int32),
    )


def preprocess(input_csv: str, output_npz: str, meta_path: str, scaler_path: str = "artifacts/scalers.joblib"):
    raw = pd.read_csv(input_csv)
    hourly = hourly_resample(raw)
    repaired, _ = repair_and_scale(hourly, scaler_path)
    X, y, ends, devices = make_sequences(repaired)
    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, X=X, y=y, ends=ends, devices=devices)
    meta = {
        "features": FEATURES,
        "lookback_hours": LOOKBACK_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "n_sequences": int(len(X)),
        "n_devices": int(repaired["device_id"].nunique()),
        "hourly_rows": int(len(repaired)),
        "target_positive_rate": float(y.mean()),
    }
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return X, y, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/iot_sensors_stream.csv")
    p.add_argument("--output", default="artifacts/preprocessed.npz")
    p.add_argument("--meta", default="artifacts/preprocess_meta.json")
    p.add_argument("--scaler", default="artifacts/scalers.joblib")
    args = p.parse_args()
    preprocess(args.input, args.output, args.meta, args.scaler)
