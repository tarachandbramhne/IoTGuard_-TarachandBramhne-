from __future__ import annotations

import numpy as np

FEATURES = ["vibration", "temperature", "pressure"]


def lightweight_attribution(sequence: np.ndarray, risk: float) -> dict:
    """Fallback attribution used when TensorFlow/SHAP artifacts are unavailable.

    It is intentionally transparent and marked as a heuristic in the API.
    """
    x = np.asarray(sequence, dtype=float)
    z = np.mean(np.abs(x - np.mean(x, axis=0, keepdims=True)), axis=0)
    total = float(z.sum()) or 1.0
    values = {f: float(v / total) for f, v in zip(FEATURES, z)}
    top = max(values, key=values.get)
    return {"method": "heuristic_demo", "risk": float(risk), "features": values, "top_sensor": top}


def shap_attribution(model, background: np.ndarray, sequence: np.ndarray) -> dict:
    import shap

    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 2:
        seq = seq[None, ...]
    bg = np.asarray(background, dtype=np.float32)
    explainer = shap.GradientExplainer(model, bg)
    raw = explainer.shap_values(seq)
    arr = raw[0] if isinstance(raw, list) else raw
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[..., 0]
    channel = np.abs(arr).mean(axis=1)[0]
    total = float(channel.sum()) or 1.0
    values = {f: float(v / total) for f, v in zip(FEATURES, channel)}
    top = max(values, key=values.get)
    return {"method": "shap_gradient", "features": values, "top_sensor": top}
