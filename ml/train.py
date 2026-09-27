"""Train the assignment LSTM failure forecaster with TensorFlow/Keras."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required for the real LSTM. Install ml/requirements.txt or run the Docker ML service."
    ) from exc

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)


def build_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="iotguard_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def chronological_split(X, y, ends, devices, train_ratio=0.8):
    order = np.argsort(ends.astype(str))
    cut = int(len(order) * train_ratio)
    tr, va = order[:cut], order[cut:]
    return (X[tr], y[tr], ends[tr], devices[tr]), (X[va], y[va], ends[va], devices[va])


def train(npz_path: str, meta_path: str, model_path: str, epochs: int = 20, batch_size: int = 128):
    data = np.load(npz_path, allow_pickle=True)
    X, y, ends, devices = data["X"], data["y"], data["ends"], data["devices"]
    (X_train, y_train, _, _), (X_val, y_val, _, _) = chronological_split(X, y, ends, devices)

    # Class weighting keeps rare failure examples visible if a different generator is used.
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    class_weight = {0: 1.0, 1: negatives / positives}

    model = build_model(X.shape[1:])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    t0 = time.perf_counter()
    prob = model.predict(X_val, batch_size=batch_size, verbose=0).reshape(-1)
    latency_ms = (time.perf_counter() - t0) * 1000 / max(1, len(X_val))
    pred = (prob >= 0.5).astype(int)

    def safe(metric, *args):
        try:
            return float(metric(*args))
        except ValueError:
            return None

    metrics = {
        "roc_auc": safe(roc_auc_score, y_val, prob),
        "pr_auc": safe(average_precision_score, y_val, prob),
        "accuracy": float(accuracy_score(y_val, pred)),
        "mae_probability": float(mean_absolute_error(y_val, prob)),
        "brier_score": float(brier_score_loss(y_val, prob)),
        "latency_ms_per_sample": float(latency_ms),
        "train_sequences": int(len(X_train)),
        "validation_sequences": int(len(X_val)),
        "epochs_run": len(history.history.get("loss", [])),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    Path(meta_path).write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return model, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/preprocessed.npz")
    p.add_argument("--meta", default="artifacts/train_metrics.json")
    p.add_argument("--model", default="artifacts/iotguard_lstm.keras")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    train(args.input, args.meta, args.model, args.epochs, args.batch_size)
