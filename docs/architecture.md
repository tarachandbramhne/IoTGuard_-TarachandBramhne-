# IoTGuard Architecture

```text
Synthetic CSV / live sensors
          │
          ▼
 POST /api/sensors/ingest
          │
          ▼
   Node / Express gateway ───── Socket.IO ───► React dashboard
          │
          ├────────────► MongoDB time-series
          ├────────────► Redis metadata/cache
          │
          ▼
      FastAPI ML service
          │
          ├── 24h feature window
          ├── Keras LSTM probability
          └── SHAP sensor attribution
```

## Deployment modes

**Developer mode:** Node/React can run against localhost services, while FastAPI can use `demo_fallback` or `heuristic` mode.

**Assignment mode:** Install TensorFlow and run `ml/preprocess.py`, `ml/train.py`, and `ml/shap_explain.py`; point the ML container/API at the resulting `.keras` model.

**Production scale:** run MongoDB as a sharded cluster keyed by `device_id`, add Redis HA, put the React app behind a CDN, and move the stream simulator to an actual MQTT/OPC-UA/Kafka ingress.
