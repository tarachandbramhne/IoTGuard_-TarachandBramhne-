from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .shap_service import FEATURES, lightweight_attribution, shap_attribution

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

APP_ROOT = Path(os.getenv("APP_ROOT", "."))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/iotguard_lstm.keras"))
FALLBACK_MODEL_PATH = Path(os.getenv("FALLBACK_MODEL_PATH", "artifacts/demo_risk_model.joblib"))
HIGH = float(os.getenv("HIGH_RISK_THRESHOLD", "0.80"))
MEDIUM = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.55"))

app = FastAPI(title="IoTGuard ML API", version="1.0.0")
model = None
model_kind = "none"


class Reading(BaseModel):
    device_id: int
    timestamp: str
    vibration: float
    temperature: float
    pressure: float
    status: str | None = None


class PredictRequest(BaseModel):
    device_id: int
    readings: list[Reading] = Field(min_length=1)
    horizon: int = Field(default=24, ge=1, le=24)


def load_models():
    global model, model_kind
    if tf is not None and MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        model_kind = "lstm"
    elif joblib is not None and FALLBACK_MODEL_PATH.exists():
        model = joblib.load(FALLBACK_MODEL_PATH)
        model_kind = "demo_fallback"
    else:
        model = None
        model_kind = "heuristic"


@app.on_event("startup")
def startup_event():
    load_models()


def feature_matrix(req: PredictRequest) -> np.ndarray:
    rows = sorted(req.readings, key=lambda r: r.timestamp)[-24:]
    values = np.array([[r.vibration, r.temperature, r.pressure] for r in rows], dtype=np.float32)
    if len(values) < 24:
        pad = np.repeat(values[:1], 24 - len(values), axis=0)
        values = np.vstack([pad, values])
    return values


def heuristic_risk(x: np.ndarray) -> float:
    vib, temp, pressure = x[-1]
    # Transparent normalized risk heuristic for demo mode.
    score = 0.50
    score += max(0.0, (vib - 15.0) / 30.0)
    score += max(0.0, (temp - 43.0) / 30.0)
    score += max(0.0, (98.0 - pressure) / 30.0)
    return float(np.clip(score, 0.01, 0.99))


def predict_risk(x: np.ndarray) -> tuple[float, str]:
    if model_kind == "lstm":
        prob = float(model.predict(x[None, ...], verbose=0).reshape(-1)[0])
        return prob, "lstm"
    if model_kind == "demo_fallback":
        # Demo model expects one flattened row of aggregate statistics.
        features = np.concatenate([x.mean(axis=0), x.max(axis=0), x.min(axis=0)])
        prob = float(model.predict_proba(features[None, :])[0, 1])
        return prob, "demo_fallback"
    return heuristic_risk(x), "heuristic"


def alert_level(risk: float) -> str:
    if risk >= HIGH:
        return "high"
    if risk >= MEDIUM:
        return "medium"
    return "normal"


@app.get("/health")
def health():
    return {"status": "ok", "model_kind": model_kind, "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(req: PredictRequest):
    x = feature_matrix(req)
    risk, backend = predict_risk(x)
    if model_kind == "lstm":
        try:
            background = np.repeat(x[None, ...], 4, axis=0)
            explanation = shap_attribution(model, background, x)
        except Exception as exc:
            explanation = lightweight_attribution(x, risk)
            explanation["shap_error"] = str(exc)
    else:
        explanation = lightweight_attribution(x, risk)
    probs = [float(np.clip(risk * (0.98 ** i), 0, 1)) for i in range(req.horizon)]
    return {
        "device_id": req.device_id,
        "horizon_hours": req.horizon,
        "probabilities": probs,
        "risk": risk,
        "alert": alert_level(risk) != "normal",
        "severity": alert_level(risk),
        "backend": backend,
        "explanation": explanation,
    }


@app.post("/explain")
def explain(req: PredictRequest):
    x = feature_matrix(req)
    risk, backend = predict_risk(x)
    if model_kind == "lstm":
        try:
            explanation = shap_attribution(model, np.repeat(x[None, ...], 4, axis=0), x)
        except Exception as exc:
            explanation = lightweight_attribution(x, risk)
            explanation["shap_error"] = str(exc)
    else:
        explanation = lightweight_attribution(x, risk)
    return {"device_id": req.device_id, "risk": risk, "backend": backend, "explanation": explanation}
