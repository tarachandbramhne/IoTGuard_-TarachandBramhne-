# Runbook

## Reset local artifacts

```bash
rm -f data/iot_sensors_stream.csv artifacts/preprocessed.npz artifacts/*.joblib artifacts/*.keras
```

## Rebuild pipeline

```bash
python ml/generate_data.py --mode month
python ml/preprocess.py
python ml/train.py
python ml/shap_explain.py
```

## Verify APIs

```bash
curl http://localhost:8000/health
curl http://localhost:4000/health
curl http://localhost:4000/api/devices
```

## Alert tuning

Use `HIGH_RISK_THRESHOLD` and `MEDIUM_RISK_THRESHOLD`. Keep thresholds conservative because false positives have operational cost, as highlighted in the assignment. fileciteturn0file0L174-L178
