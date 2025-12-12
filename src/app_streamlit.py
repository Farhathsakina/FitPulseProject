import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parents[2]
KAGGLE_DIR = BASE / "kaggle_data"
RAW = BASE / "data_raw"
PROC = BASE / "data_processed"
RAW.mkdir(exist_ok=True, parents=True)
PROC.mkdir(exist_ok=True, parents=True)

FILES = {
    "hr_in": KAGGLE_DIR / "heartrate_seconds.csv",
    "steps_in": KAGGLE_DIR / "minuteStepsNarrow.csv",
    "sleep_in": KAGGLE_DIR / "sleepDay_merged.csv",
}
OUT = {
    "hr": RAW / "heart_rate.csv",
    "steps": RAW / "steps.csv",
    "sleep": RAW / "sleep.csv",
    "clean": PROC / "cleaned_fitness_data.csv"
}

def try_read(p):
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def normalize_hr(df):
    if df.empty:
        return pd.DataFrame(columns=["bpm"])
    ts = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), df.columns[1] if len(df.columns)>1 else df.columns[0])
    val = next((c for c in df.columns if c.lower() in ("value","bpm","heartrate")), df.columns[-1])
    out = pd.DataFrame({"timestamp": pd.to_datetime(df[ts], errors="coerce"), "bpm": pd.to_numeric(df[val], errors="coerce")})
    return out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

def normalize_steps(df):
    if df.empty:
        return pd.DataFrame(columns=["steps"])
    ts = next((c for c in df.columns if "activity" in c.lower() or "time" in c.lower()), df.columns[0])
    val = next((c for c in df.columns if "step" in c.lower()), df.columns[-1])
    out = pd.DataFrame({"timestamp": pd.to_datetime(df[ts], errors="coerce"), "steps": pd.to_numeric(df[val], errors="coerce").fillna(0).astype(int)})
    return out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

def save_separate_files(hr, stp, sl):
    pairs = [(hr, OUT["hr"]), (stp, OUT["steps"]), (sl, OUT["sleep"])]
    for df, path in pairs:
        if df is None or df.empty:
            continue
        df_reset = df.reset_index().rename(columns={"index": "timestamp"})
        df_reset.to_csv(path, index=False)

def merge_and_resample(hr, stp, freq="1min"):
    # hr, stp are DataFrames with datetime index (or empty)
    if (hr is None or hr.empty) and (stp is None or stp.empty):
        return pd.DataFrame()
    # ensure unique timestamps by grouping (avoid duplicate-label reindex errors)
    if not hr.empty:
        hr = hr.reset_index().rename(columns={"index":"timestamp"})
        hr['timestamp'] = pd.to_datetime(hr['timestamp'], errors='coerce')
        hr = hr.dropna(subset=['timestamp'])
        hr = hr.groupby('timestamp', as_index=True)['bpm'].mean().sort_index()
    else:
        hr = pd.Series(dtype=float)

    if not stp.empty:
        stp = stp.reset_index().rename(columns={"index":"timestamp"})
        stp['timestamp'] = pd.to_datetime(stp['timestamp'], errors='coerce')
        stp = stp.dropna(subset=['timestamp'])
        stp = stp.groupby('timestamp', as_index=True)['steps'].sum().sort_index()
    else:
        stp = pd.Series(dtype=int)

    # build continuous 1-minute index from available data
    min_ts = min([x.index.min() for x in (hr, stp) if not x.empty])
    max_ts = max([x.index.max() for x in (hr, stp) if not x.empty])
    idx = pd.date_range(start=min_ts.floor('T'), end=max_ts.ceil('T'), freq=freq)

    hr_df = hr.to_frame().reindex(idx).rename_axis("timestamp")
    st_df = stp.to_frame().reindex(idx).rename_axis("timestamp")

    # fill/interpolate
    if 'bpm' in hr_df.columns:
        hr_df['bpm'] = hr_df['bpm'].interpolate(method='time').ffill().bfill()
    else:
        hr_df['bpm'] = np.nan
    if 'steps' in st_df.columns:
        st_df['steps'] = st_df['steps'].fillna(0).astype(int)
    else:
        st_df['steps'] = 0

    merged = pd.concat([hr_df['bpm'], st_df['steps']], axis=1).reset_index()
    merged.to_csv(OUT["clean"], index=False)
    return merged

# ---------- UI ----------
st.set_page_config(page_title="FitPulse — Preprocessing Dashboard", layout="wide")
st.title("FitPulse — Data Collection & Cleaning Dashboard")

st.write("Click the button below to load & preprocess Kaggle Fitbit files from `kaggle_data/`.")

if st.button("Run preprocessing and write CSV files"):
    st.info("Reading raw files...")
    hr_raw = try_read(FILES["hr_in"])
    steps_raw = try_read(FILES["steps_in"])
    sleep_raw = try_read(FILES["sleep_in"])

    st.info("Normalizing heart rate...")
    hr = normalize_hr(hr_raw)

    st.info("Normalizing steps...")
    stp = normalize_steps(steps_raw)

    st.info("Saving separate CSV outputs...")
    save_separate_files(hr, stp, sleep_raw)
    st.success("Separate files written to data_raw/")

    st.info("Merging & resampling to 1-minute...")
    merged = merge_and_resample(hr, stp)
    st.success("Merged file saved to data_processed/")

    # Robust preview + plotting
    st.subheader("Preview of Cleaned Merged Data")
    if merged.empty:
        st.warning("Merged data is empty — ensure the kaggle_data files exist and contain timestamps.")
    else:
        # ensure timestamp column exists
        if 'timestamp' not in merged.columns:
            merged = merged.reset_index().rename(columns={merged.index.name or 'index': 'timestamp'})
        merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
        if merged['timestamp'].isna().all():
            st.warning("Could not parse timestamps. Open cleaned CSV to inspect.")
            st.dataframe(merged.head(10))
        else:
            st.dataframe(merged.head(100))
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            if 'bpm' in merged.columns:
                axes[0].plot(merged['timestamp'], merged['bpm'])
                axes[0].set_title("Heart Rate (bpm)")
            else:
                axes[0].text(0.5, 0.5, "No bpm column", ha='center')
            if 'steps' in merged.columns:
                axes[1].plot(merged['timestamp'], merged['steps'])
                axes[1].set_title("Steps per minute")
            else:
                axes[1].text(0.5, 0.5, "No steps column", ha='center')
            st.pyplot(fig)
            st.download_button("Download cleaned CSV",
                               data=merged.to_csv(index=False).encode('utf-8'),
                               file_name="cleaned_fitness_data.csv",
                               mime="text/csv")

st.markdown("Files expected in project: `kaggle_data/heartrate_seconds.csv`, `kaggle_data/minuteStepsNarrow.csv`, `kaggle_data/sleepDay_merged.csv`")
