from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path.home() / "FitPulseProject"
RAW = BASE / "data_raw"
PROC = BASE / "data_processed"
PROC.mkdir(parents=True, exist_ok=True)

HR = RAW / "heart_rate.csv"
STEPS = RAW / "steps.csv"
OUT = PROC / "cleaned_fitness_data.csv"

def safe_read(p):
    try:
        return pd.read_csv(p, parse_dates=["timestamp"])
    except Exception:
        return pd.read_csv(p)

print("Reading:", HR if HR.exists() else "MISSING", STEPS if STEPS.exists() else "MISSING")

# Read and aggregate duplicates
hr = safe_read(HR) if HR.exists() else pd.DataFrame(columns=["timestamp","bpm"])
st = safe_read(STEPS) if STEPS.exists() else pd.DataFrame(columns=["timestamp","steps"])

if not hr.empty:
    # ensure timestamp parsed
    hr['timestamp'] = pd.to_datetime(hr['timestamp'], errors='coerce')
    # drop bad rows
    hr = hr.dropna(subset=['timestamp'])
    # group by timestamp and average bpm (handles duplicate labels)
    hr = hr.groupby('timestamp', as_index=True)['bpm'].mean().sort_index()
else:
    hr = pd.Series(dtype=float)

if not st.empty:
    st['timestamp'] = pd.to_datetime(st['timestamp'], errors='coerce')
    st = st.dropna(subset=['timestamp'])
    # group by timestamp and sum steps (if duplicates)
    st = st.groupby('timestamp', as_index=True)['steps'].sum().sort_index()
else:
    st = pd.Series(dtype=int)

if hr.empty and st.empty:
    print("No data available to merge. Exiting.")
else:
    # build combined index
    min_ts = min([x.index.min() for x in (hr,st) if not x.empty])
    max_ts = max([x.index.max() for x in (hr,st) if not x.empty])
    idx = pd.date_range(start=min_ts.floor('T'), end=max_ts.ceil('T'), freq='1min')
    # reindex
    hr_r = hr.reindex(idx).rename_axis("timestamp").to_frame()
    st_r = st.reindex(idx).rename_axis("timestamp").to_frame()
    # interpolate/ffill/bfill hr, fill steps with 0
    if 'bpm' in hr_r.columns:
        hr_r['bpm'] = hr_r['bpm'].interpolate(method='time').ffill().bfill()
    else:
        hr_r['bpm'] = np.nan
    if 'steps' in st_r.columns:
        st_r['steps'] = st_r['steps'].fillna(0).astype(int)
    else:
        st_r['steps'] = 0
    merged = pd.concat([hr_r['bpm'], st_r['steps']], axis=1).reset_index()
    merged.to_csv(OUT, index=False)
    print("Wrote merged file:", OUT, "rows:", len(merged))
