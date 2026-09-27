from fastapi.testclient import TestClient
from ml.app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict():
    readings = []
    for i in range(24):
        readings.append({
            'device_id': 1,
            'timestamp': f'2024-01-01T{i:02d}:00:00',
            'vibration': 20.0 + i * .1,
            'temperature': 45.0,
            'pressure': 96.0,
        })
    r = client.post('/predict', json={'device_id':1,'readings':readings,'horizon':24})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body['risk'] <= 1
    assert len(body['probabilities']) == 24
    assert 'explanation' in body
