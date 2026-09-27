from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/iot_sensors_stream.csv');p.add_argument('--url',default='http://localhost:4000/api/sensors/ingest');p.add_argument('--rate',type=int,default=1000);p.add_argument('--batch',type=int,default=25);a=p.parse_args()
    df=pd.read_csv(a.csv); delay=max(0.001, a.batch/a.rate)
    rows=df.to_dict('records')
    for i in range(0,len(rows),a.batch):
        batch=rows[i:i+a.batch]
        r=requests.post(a.url,json=batch,timeout=20);r.raise_for_status()
        print(f'sent {i+len(batch)}/{len(rows)}')
        time.sleep(delay)
if __name__=='__main__': main()
