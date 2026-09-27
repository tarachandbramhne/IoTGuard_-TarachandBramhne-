"""Train a tiny scikit-learn fallback so the UI can run without TensorFlow.
This is a demo backend only; the official assignment model is ml/train.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def prepare(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    rows = []
    for device, g in df.groupby("device_id"):
        g = g.sort_values("timestamp").set_index("timestamp")
        h = g[["vibration","temperature","pressure","status"]].copy()
        agg = h[["vibration","temperature","pressure"]].resample("1h").mean().ffill().bfill()
        fail = h["status"].eq("fail").resample("1h").max().fillna(False).astype(int)
        for i in range(24, len(agg) - 24 + 1):
            x = agg.iloc[i-24:i]
            y = int(fail.iloc[i:i+24].max())
            rows.append((np.concatenate([x.mean().to_numpy(), x.max().to_numpy(), x.min().to_numpy()]), y))
    X = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/iot_sensors_stream.csv');p.add_argument('--output',default='artifacts/demo_risk_model.joblib');a=p.parse_args()
    df=pd.read_csv(a.input);X,y=prepare(df)
    model=LogisticRegression(max_iter=500,class_weight='balanced');model.fit(X,y)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,a.output)
    print('Saved fallback:',a.output,'samples=',len(y),'positive_rate=',float(y.mean()))
if __name__=='__main__': main()
