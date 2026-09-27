import pandas as pd
import numpy as np
from ml.preprocess import hourly_resample, repair_and_scale, make_sequences

def test_hourly_resample_and_sequence_creation():
    ts = pd.date_range('2024-01-01', periods=120, freq='30min')
    df = pd.DataFrame({
        'device_id': 1,
        'timestamp': ts.astype(str),
        'vibration': np.linspace(10, 20, len(ts)),
        'temperature': np.linspace(40, 45, len(ts)),
        'pressure': np.linspace(100, 95, len(ts)),
        'status': ['normal'] * 118 + ['fail', 'fail'],
    })
    hourly = hourly_resample(df)
    repaired, _ = repair_and_scale(hourly)
    # Add enough hours for a single 24+24 sequence in this focused unit test.
    repaired = pd.concat([repaired] * 4, ignore_index=True)
    repaired['timestamp'] = pd.date_range('2024-01-01', periods=len(repaired), freq='1h')
    repaired['device_id'] = 1
    X, y, _, _ = make_sequences(repaired)
    assert X.shape[1:] == (24, 3)
    assert set(np.unique(y)).issubset({0.0, 1.0})
