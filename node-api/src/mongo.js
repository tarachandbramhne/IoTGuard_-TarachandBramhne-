import { MongoClient } from 'mongodb';

export async function connectMongo(uri) {
  const client = new MongoClient(uri);
  await client.connect();
  const db = client.db();
  try {
    await db.createCollection('sensor_readings', {
      timeseries: { timeField: 'timestamp', metaField: 'device_id', granularity: 'minutes' },
      expireAfterSeconds: 60 * 60 * 24 * 90
    });
  } catch (_) {}
  const sensors = db.collection('sensor_readings');
  try { await sensors.createIndex({ device_id: 1, timestamp: -1 }); } catch (_) {}
  return { client, db, sensors };
}
