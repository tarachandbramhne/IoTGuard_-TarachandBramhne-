"""Generate SHAP explanations for TensorFlow LSTM sequences."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf

FEATURES = ["vibration", "temperature", "pressure"]


def explain(model_path: str, npz_path: str, output_dir: str, background_size: int = 100, explain_size: int = 20):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    background = X[:background_size]
    samples = X[:explain_size]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(samples)
    arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[..., 0]
    # Aggregate over time for each sensor.
    feature_abs = np.abs(arr).mean(axis=1).mean(axis=0)
    importance = dict(zip(FEATURES, feature_abs.tolist()))

    # A compact SHAP summary across sensor channels.
    flattened = arr.reshape(arr.shape[0], -1)
    names = [f"{f}_t{t+1}" for t in range(arr.shape[1]) for f in FEATURES]
    shap.summary_plot(
        flattened,
        features=samples.reshape(samples.shape[0], -1),
        feature_names=names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    lines = "\n".join(f"{k}: {v:.6f}" for k, v in sorted(importance.items(), key=lambda x: -x[1]))
    Path(output_dir, "feature_importance.txt").write_text(lines, encoding="utf-8")
    print(lines)
    return importance


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/iotguard_lstm.keras")
    p.add_argument("--input", default="artifacts/preprocessed.npz")
    p.add_argument("--output-dir", default="artifacts/shap")
    args = p.parse_args()
    explain(args.model, args.input, args.output_dir)
