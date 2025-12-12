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

hr = safe_read(HR) if HR.exists() else pd.DataFrame(columns=["timestamp","bpm"])
st = safe_read(STEPS) if STEPS.exists() else pd.DataFrame(columns=["timestamp","steps"])

# normalize & drop bad rows
if not hr.empty:
    hr['timestamp'] = pd.to_datetime(hr['timestamp'], errors='coerce')
    hr = hr.dropna(subset=['timestamp'])
    # aggregate duplicates by mean
    hr = hr.groupby('timestamp', as_index=True)['bpm'].mean().sort_index()
    print("Heart rate points after grouping (unique timestamps):", len(hr))
else:
    hr = pd.Series(dtype=float)

if not st.empty:
    st['timestamp'] = pd.to_datetime(st['timestamp'], errors='coerce')
    st = st.dropna(subset=['timestamp'])
    # aggregate duplicates by sum (steps)
    st = st.groupby('timestamp', as_index=True)['steps'].sum().sort_index()
    print("Steps points after grouping (unique timestamps):", len(st))
else:
    st = pd.Series(dtype=int)

if hr.empty and st.empty:
    print("No data to merge. Exiting.")
else:
    # build index safely
    min_ts = min([x.index.min() for x in (hr,st) if not x.empty])
    max_ts = max([x.index.max() for x in (hr,st) if not x.empty])
    idx = pd.date_range(start=min_ts.floor('T'), end=max_ts.ceil('T'), freq='1min')

    # to_frame ensures we reindex DataFrame columns (safe)
    hr_df = hr.to_frame().reindex(idx).rename_axis("timestamp")
    st_df = st.to_frame().reindex(idx).rename_axis("timestamp")

    # now there cannot be duplicate labels on the index
    # interpolate HR and fill steps
    if 'bpm' in hr_df.columns:
        hr_df['bpm'] = hr_df['bpm'].interpolate(method='time').ffill().bfill()
    else:
        hr_df['bpm'] = np.nan

    if 'steps' in st_df.columns:
        st_df['steps'] = st_df['steps'].fillna(0).astype(int)
    else:
        st_df['steps'] = 0

    merged = pd.concat([hr_df['bpm'], st_df['steps']], axis=1).reset_index()
    merged.to_csv(OUT, index=False)
    print("Wrote merged file:", OUT, "rows:", len(merged))
