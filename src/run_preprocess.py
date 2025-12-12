from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path.home() / "FitPulseProject"
KAG = BASE / "kaggle_data"
RAW = BASE / "data_raw"
PROC = BASE / "data_processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# inputs (we copied these earlier)
HR_IN = KAG / "heartrate_seconds.csv"
STEPS_IN = KAG / "minuteStepsNarrow.csv"
SLEEP_IN = KAG / "sleepDay_merged.csv"

HR_OUT = RAW / "heart_rate.csv"
STEPS_OUT = RAW / "steps.csv"
SLEEP_OUT = RAW / "sleep.csv"
CLEAN_OUT = PROC / "cleaned_fitness_data.csv"

def safe_read(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print("Failed to read", p, ":", e)
        return pd.DataFrame()

print("Loading inputs...")
hr_raw = safe_read(HR_IN)
steps_raw = safe_read(STEPS_IN)
sleep_raw = safe_read(SLEEP_IN)

# normalize heart rate (Kaggle format: Id, Time, Value)
if not hr_raw.empty:
    # try several possible timestamp/value columns
    ts_col = next((c for c in hr_raw.columns if "time" in c.lower() or "date" in c.lower()), hr_raw.columns[1])
    val_col = next((c for c in hr_raw.columns if c.lower() in ("value","bpm","heartrate")), hr_raw.columns[-1])
    hr = pd.DataFrame({
        "timestamp": pd.to_datetime(hr_raw[ts_col], errors="coerce"),
        "bpm": pd.to_numeric(hr_raw[val_col], errors="coerce")
    }).dropna(subset=["timestamp"])
    hr = hr.set_index("timestamp").sort_index()
    hr.reset_index().to_csv(HR_OUT, index=False)
    print("Wrote", HR_OUT, "rows:", len(hr))
else:
    hr = pd.DataFrame(columns=["bpm"])
    print("No heart rate input")

# normalize steps (Kaggle: ActivityMinute, Steps)
if not steps_raw.empty:
    ts = next((c for c in steps_raw.columns if "activity" in c.lower() or "time" in c.lower()), steps_raw.columns[0])
    val = next((c for c in steps_raw.columns if "step" in c.lower()), steps_raw.columns[-1])
    st = pd.DataFrame({
        "timestamp": pd.to_datetime(steps_raw[ts], errors="coerce"),
        "steps": pd.to_numeric(steps_raw[val], errors="coerce").fillna(0).astype(int)
    }).dropna(subset=["timestamp"])
    st = st.set_index("timestamp").sort_index()
    st.reset_index().to_csv(STEPS_OUT, index=False)
    print("Wrote", STEPS_OUT, "rows:", len(st))
else:
    st = pd.DataFrame(columns=["steps"])
    print("No steps input")

# save sleep raw copy (session-level)
if not sleep_raw.empty:
    try:
        sleep_raw.to_csv(SLEEP_OUT, index=False)
        print("Wrote", SLEEP_OUT, "rows:", len(sleep_raw))
    except Exception as e:
        print("Failed to write sleep:", e)

# merge & resample to 1-minute
if (not hr.empty) or (not st.empty):
    # determine index range from what exists
    min_ts = min([x.index.min() for x in (hr,st) if not x.empty])
    max_ts = max([x.index.max() for x in (hr,st) if not x.empty])
    idx = pd.date_range(start=min_ts.floor('T'), end=max_ts.ceil('T'), freq='1T')
    hr_r = hr.reindex(idx).rename_axis("timestamp")
    st_r = st.reindex(idx).rename_axis("timestamp")
    if 'bpm' in hr_r.columns:
        hr_r['bpm'] = hr_r['bpm'].interpolate(method='time').ffill().bfill()
    else:
        hr_r['bpm'] = np.nan
    if 'steps' in st_r.columns:
        st_r['steps'] = st_r['steps'].fillna(0).astype(int)
    else:
        st_r['steps'] = 0
    merged = pd.concat([hr_r['bpm'], st_r['steps']], axis=1).reset_index()
    merged.to_csv(CLEAN_OUT, index=False)
    print("Wrote merged cleaned file:", CLEAN_OUT, "rows:", len(merged))
else:
    print("No data to merge; skipping merged output")
