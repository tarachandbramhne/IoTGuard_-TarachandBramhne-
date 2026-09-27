# IoTGuard – Predictive Maintenance for IoT Devices

A full-stack reference implementation of the **IoTGuard predictive maintenance assignment**: synthetic IoT sensor generation, gap handling, hourly resampling, 24-hour sequence construction, Keras/TensorFlow LSTM forecasting, SHAP explainability, FastAPI ML service, Node/Express + Socket.IO ingestion, MongoDB/Redis persistence, and a React monitoring dashboard.

The implementation follows the supplied assignment requirements: LSTM-based failure forecasting, 24-hour lookback/forecast horizon, SHAP explanations, REST APIs, WebSocket updates, MongoDB time-series persistence, and a React device grid. fileciteturn0file0L17-L38

## Important source-data note

The assignment's generator says that 2,000 samples per device represent about one month, but the supplied timestamp formula produces approximately 0.0463 minutes between samples (about 92.6 minutes total per device). The project therefore provides two modes:

- `--mode assignment`: reproduces the supplied timestamp formula as closely as possible.
- `--mode month`: uses `21.6` minutes/sample so 2,000 readings span roughly 30 days. In this runnable demo mode, the `10%` failure rate is interpreted as approximately 10% of devices receiving clustered failure events; this keeps the `next 24h` target from becoming almost always positive. The supplied per-reading failure formula remains available in assignment mode. fileciteturn0file0L44-L63

## Repository layout

```text
IoTGuard_Full_Project/
├── data/                       # generated CSVs / local demo data
├── ml/                         # simulator, preprocessing, LSTM, SHAP, FastAPI
├── api/                        # FastAPI Docker entrypoint
├── node-api/                   # Express + Socket.IO ingestion/gateway
├── frontend/                  # React + Recharts dashboard
├── artifacts/                 # generated model/scaler/metadata files
├── tests/                     # preprocessing/API tests
├── scripts/                   # stream simulator and local helpers
├── docs/                      # architecture and runbook
├── docker-compose.yml
├── .env.example
└── README.md
```

## Assignment-to-implementation map

| Requirement | Implementation |
|---|---|
| Simulate 50 devices, sensor failures and gaps | `ml/generate_data.py` |
| Forward-fill + KNN imputation | `ml/preprocess.py` |
| Hourly resampling | `ml/preprocess.py` |
| 24-hour overlapping sequences | `ml/preprocess.py` |
| Keras/TensorFlow multi-layer LSTM | `ml/train.py` |
| Failure probability for next 24h | `ml/train.py`, `ml/app/main.py` |
| SHAP explanation | `ml/shap_explain.py`, `ml/app/shap_service.py` |
| Prediction API | FastAPI `/predict`, Node `/api/predict/:deviceId` |
| Sensor ingestion | Node `POST /api/sensors/ingest` |
| WebSockets | Socket.IO `/realtime` |
| MongoDB persistence | `node-api/src/mongo.js` |
| Redis device metadata cache | `node-api/src/server.js` |
| Device risk grid | `frontend/src/App.jsx` |
| Trend chart | Recharts in `frontend/src/App.jsx` |
| SHAP bar chart | Recharts in `frontend/src/App.jsx` |
| Cypress smoke test | `frontend/cypress/e2e/dashboard.cy.js` |

The assignment calls for preprocessing, LSTM forecasting, SHAP, REST endpoints, WebSockets, a React grid, and tests/benchmarks. fileciteturn0file0L109-L157

## Quick start – local Python demo

This creates a working dataset and a lightweight scikit-learn fallback artifact so the dashboard can be exercised even before TensorFlow is installed. The fallback is **not** the production LSTM; it is clearly labelled `demo_fallback` in API responses. The official LSTM trainer remains `ml/train.py`.

### 1. Create a Python environment

With Anaconda or Miniconda:

```bash
conda env create -f environment.yml
conda activate iotguard
```

Or use `venv`:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r ml/requirements.txt
```

### 2. Generate data

```bash
python ml/generate_data.py --mode month --output data/iot_sensors_stream.csv
```

The assignment specifies 50 devices × 2,000 readings/device, 5% missing rows and a 10% failure mechanism. fileciteturn0file0L44-L78

### 3. Preprocess

```bash
python ml/preprocess.py \
  --input data/iot_sensors_stream.csv \
  --output artifacts/preprocessed.npz \
  --meta artifacts/preprocess_meta.json
```

### 4. Train the LSTM (when TensorFlow is available)

```bash
python ml/train.py \
  --input artifacts/preprocessed.npz \
  --meta artifacts/preprocess_meta.json \
  --model artifacts/iotguard_lstm.keras
```

The model uses a multi-layer LSTM with dropout and a binary output. The assignment explicitly requests this pattern. fileciteturn0file0L109-L117

### 5. Create a demo fallback model (optional but useful for immediate API/UI smoke testing)

```bash
python scripts/train_demo_fallback.py \
  --input data/iot_sensors_stream.csv \
  --output artifacts/demo_risk_model.joblib
```

### 6. Start ML service

```bash
uvicorn ml.app.main:app --host 0.0.0.0 --port 8000
```

### 7. Start Node gateway

```bash
cd node-api
npm install
npm run dev
```

### 8. Start React dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Docker Compose

The Compose stack provides MongoDB, Redis, the ML API, Node gateway, and React frontend. TensorFlow is installed inside the ML container, so the host environment does not need TensorFlow.

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:5173`
- Node API: `http://localhost:4000`
- ML API: `http://localhost:8000/docs`
- MongoDB: `mongodb://localhost:27017`
- Redis: `redis://localhost:6379`

The assignment recommends Docker for MongoDB/Redis and Python training. fileciteturn0file0L148-L156

## Streaming simulation

To send synthetic readings into the gateway at up to 1,000 readings/minute:

```bash
python scripts/stream_simulator.py \
  --csv data/iot_sensors_stream.csv \
  --url http://localhost:4000/api/sensors/ingest \
  --rate 1000
```

The assignment asks for a stream simulation at 1,000 readings/minute and WebSocket forecast updates. fileciteturn0file0L140-L147

## API

### Node gateway

- `POST /api/sensors/ingest` – accepts an array of sensor readings.
- `GET /api/predict/:deviceId?horizon=24` – forecast and SHAP explanation.
- `GET /api/alerts?severity=high&page=1&pageSize=50` – recent alerts.
- `GET /api/devices` – device risk cards.
- `GET /api/devices/:deviceId/trends?hours=24` – sensor trends.
- `GET /health` – service health.
- Socket.IO namespace/path: `/realtime`.

The paths are aligned to the assignment's stated API/Socket contract. fileciteturn0file0L118-L139

### ML service

- `POST /predict` – forecast from supplied latest readings.
- `GET /health` – model/service status.
- `POST /explain` – SHAP/feature attribution for a sequence.

## SHAP

`ml/shap_explain.py` supports TensorFlow-aware SHAP via `GradientExplainer` and produces a summary plot plus feature-level importance. The runtime service also exposes a compact explanation payload where sensor contributions are grouped into `vibration`, `temperature`, and `pressure`. The assignment specifically asks engineers to be told which sensor contributes most to a high-risk prediction. fileciteturn0file0L24-L33

## MongoDB / scale notes

The local Compose deployment uses one Mongo instance for simplicity. For a 1k+ sensor production setup, configure MongoDB as a sharded cluster and shard time-series data by `device_id`; the application code already keeps `device_id` as the high-selectivity partition key. The assignment explicitly calls for sharded time-series storage. fileciteturn0file0L9-L16

## Metrics and thresholding

The LSTM trainer prints:

- ROC-AUC
- PR-AUC
- binary accuracy
- MAE on failure probability
- Brier score
- latency per batch

Because the target is a binary failure probability, MAE/RMSE on raw probabilities is the meaningful interpretation for this implementation; the assignment's “RMSE < 2 on values” wording is not directly applicable to a 0–1 binary target. This repository therefore reports the raw probability metrics and documents the distinction.

The UI uses conservative alert thresholds from environment variables:

```text
HIGH_RISK_THRESHOLD=0.80
MEDIUM_RISK_THRESHOLD=0.55
```

This supports the assignment's warning that false positives can create downtime/alert fatigue. fileciteturn0file0L15-L16

## Tests

```bash
pytest -q
```

Frontend smoke test:

```bash
cd frontend
npx cypress run
```

The assignment requests 80% coverage across ML/API tests. The included test structure is ready for coverage expansion; exact coverage will depend on the TensorFlow/Mongo services available in the environment where tests are run. fileciteturn0file0L148-L157

## Limitations and honest status

This package includes the complete source code and runnable architecture. In this environment, TensorFlow and MongoDB client packages are not installed and external package installation is unavailable, so a real `.keras` LSTM artifact cannot be trained here. A small `demo_risk_model.joblib` fallback is included for end-to-end UI/API smoke tests; it must not be presented as the assignment's LSTM model. Install the requirements or use the Docker ML container to train the actual LSTM.

The source assignment also recommends future field testing, robustness work, and explicit documentation of failure simulation. fileciteturn0file0L170-L178
