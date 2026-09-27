import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import http from 'http';
import { Server } from 'socket.io';
import axios from 'axios';
import { createClient } from 'redis';
import { connectMongo } from './mongo.js';

const PORT = Number(process.env.PORT || 4000);
const ML_URL = process.env.ML_URL || 'http://localhost:8000';
const HIGH = Number(process.env.HIGH_RISK_THRESHOLD || 0.8);
const MEDIUM = Number(process.env.MEDIUM_RISK_THRESHOLD || 0.55);
const USE_MONGO = String(process.env.USE_MONGO || 'true').toLowerCase() === 'true';

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });
const realtime = io.of('/realtime');

let sensors = null;
let mongoClient = null;
let redis = null;
const memory = new Map();

function severity(risk) {
  if (risk >= HIGH) return 'high';
  if (risk >= MEDIUM) return 'medium';
  return 'normal';
}

async function init() {
  if (USE_MONGO) {
    try {
      const conn = await connectMongo(process.env.MONGO_URI || 'mongodb://localhost:27017/iotguard');
      mongoClient = conn.client;
      sensors = conn.sensors;
    } catch (err) {
      console.warn('Mongo unavailable; using memory store:', err.message);
    }
  }
  try {
    redis = createClient({ url: process.env.REDIS_URL || 'redis://localhost:6379' });
    redis.on('error', (err) => console.warn('Redis:', err.message));
    await redis.connect();
  } catch (err) {
    console.warn('Redis unavailable; continuing without cache:', err.message);
  }
}

async function saveReadings(readings) {
  for (const row of readings) {
    const item = { ...row, device_id: Number(row.device_id), timestamp: new Date(row.timestamp) };
    if (!memory.has(item.device_id)) memory.set(item.device_id, []);
    memory.get(item.device_id).push(item);
  }
  if (sensors && readings.length) {
    await sensors.insertMany(readings.map((r) => ({
      ...r,
      device_id: Number(r.device_id),
      timestamp: new Date(r.timestamp),
    })));
  }
}

async function getLatest(deviceId, limit = 168) {
  if (sensors) {
    return sensors.find({ device_id: Number(deviceId) }).sort({ timestamp: -1 }).limit(limit).toArray().then((x) => x.reverse());
  }
  return (memory.get(Number(deviceId)) || []).slice(-limit);
}

async function predict(deviceId, readings, horizon = 24) {
  const payload = {
    device_id: Number(deviceId),
    horizon,
    readings: readings.map((r) => ({ ...r, device_id: Number(deviceId) })),
  };
  const { data } = await axios.post(`${ML_URL}/predict`, payload, { timeout: 15000 });
  return data;
}

app.get('/health', async (_, res) => {
  res.json({ status: 'ok', mongo: Boolean(sensors), redis: Boolean(redis), ml_url: ML_URL });
});

app.post('/api/sensors/ingest', async (req, res) => {
  try {
    const readings = Array.isArray(req.body) ? req.body : (req.body.readings || []);
    if (!readings.length) return res.status(400).json({ error: 'readings array is required' });
    await saveReadings(readings);
    const updates = [];
    const byDevice = new Map();
    for (const r of readings) {
      if (!byDevice.has(Number(r.device_id))) byDevice.set(Number(r.device_id), []);
      byDevice.get(Number(r.device_id)).push(r);
    }
    for (const [deviceId] of byDevice) {
      const latest = await getLatest(deviceId, 72);
      const forecast = await predict(deviceId, latest, 24);
      const update = { ...forecast, latest: latest.at(-1) };
      updates.push(update);
      realtime.emit('forecast:update', update);
      if (redis) await redis.set(`device:${deviceId}:risk`, JSON.stringify(update), { EX: 3600 });
    }
    res.json({ accepted: readings.length, updates });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/predict/:deviceId', async (req, res) => {
  try {
    const latest = await getLatest(req.params.deviceId, 72);
    if (!latest.length) return res.status(404).json({ error: 'device not found' });
    res.json(await predict(req.params.deviceId, latest, Number(req.query.horizon || 24)));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/devices/:deviceId/trends', async (req, res) => {
  const rows = await getLatest(req.params.deviceId, Number(req.query.hours || 24));
  res.json(rows.map((r) => ({
    timestamp: new Date(r.timestamp).toISOString(),
    vibration: Number(r.vibration),
    temperature: Number(r.temperature),
    pressure: Number(r.pressure),
  })));
});

app.get('/api/alerts', async (req, res) => {
  const wanted = req.query.severity || 'high';
  const pageSize = Number(req.query.pageSize || 50);
  const page = Number(req.query.page || 1);
  const alerts = [];
  for (const deviceId of memory.keys()) {
    const latest = await getLatest(deviceId, 72);
    if (!latest.length) continue;
    try {
      const forecast = await predict(deviceId, latest, 24);
      if (severity(forecast.risk) === wanted) alerts.push({ ...forecast, timestamp: new Date().toISOString() });
    } catch (_) {}
  }
  res.json({ page, pageSize, total: alerts.length, alerts: alerts.slice((page - 1) * pageSize, page * pageSize) });
});

app.get('/api/devices', async (_, res) => {
  const ids = new Set(memory.keys());
  if (sensors) {
    const docs = await sensors.distinct('device_id');
    docs.forEach((id) => ids.add(Number(id)));
  }
  const out = [];
  for (const deviceId of [...ids].sort((a, b) => a - b).slice(0, 200)) {
    const latest = await getLatest(deviceId, 72);
    if (!latest.length) continue;
    try {
      const forecast = await predict(deviceId, latest, 24);
      out.push({ device_id: deviceId, risk: forecast.risk, severity: forecast.severity, backend: forecast.backend });
    } catch (_) {}
  }
  res.json(out);
});

realtime.on('connection', (socket) => {
  socket.emit('realtime:connected', { ok: true, at: new Date().toISOString() });
});

init().then(() => server.listen(PORT, () => console.log(`IoTGuard API listening on ${PORT}`)));
process.on('SIGINT', async () => {
  if (redis) await redis.quit().catch(() => {});
  if (mongoClient) await mongoClient.close().catch(() => {});
  process.exit(0);
});
