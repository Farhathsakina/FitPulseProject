import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime

st.set_page_config(page_title="FitPulse — Sleep & Activity Dashboard", layout="wide")
st.title("FitPulse — Sleep, Activity & Cleaned Data Viewer")

BASE = Path.home() / "FitPulseProject"
CLEANED = BASE / "data_processed" / "cleaned_fitness_data.csv"
KAG = BASE / "kaggle_data"

sleep_day_candidates = [
    KAG / "sleepDay_merged.csv",
    BASE / "data_raw" / "sleep.csv"
]
minute_sleep_candidates = [
    KAG / "minuteSleep_merged.csv",
    KAG / "minuteSleep.csv",
    BASE / "data_raw" / "minuteSleep.csv"
]
daily_activity_candidates = [
    KAG / "dailyActivity_merged.csv",
    KAG / "dailyActivity.csv",
    BASE / "data_raw" / "dailyActivity.csv"
]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

sleep_day_file = first_existing(sleep_day_candidates)
minute_sleep_file = first_existing(minute_sleep_candidates)
daily_activity_file = first_existing(daily_activity_candidates)

if not CLEANED.exists():
    st.error(f"Cleaned file not found: {CLEANED}")
    st.write("Run the preprocessing script first (run_preprocess.py or merge_fix2.py).")
    st.stop()

# load cleaned data
df = pd.read_csv(CLEANED, parse_dates=["timestamp"])
st.success(f"Loaded cleaned file ({len(df)} rows).")

# ------------- Sleep daily summary ----------------
st.header("Sleep — Daily Summary")
if sleep_day_file:
    sd = pd.read_csv(sleep_day_file, parse_dates=[col for col in ["SleepDay"] if "SleepDay" in open(sleep_day_file).read(2000)])
    # normalize columns (common names)
    cols = sd.columns.str.lower()
    # Try to locate typical columns
    tot_sleep_col = next((c for c in sd.columns if "totalminutesasleep" in c.lower()), None)
    tot_bed_col = next((c for c in sd.columns if "totaltimeinbed" in c.lower()), None)
    sleep_day_col = next((c for c in sd.columns if "sleepday" in c.lower()), None)
    display_cols = []
    if sleep_day_col:
        sd[sleep_day_col] = pd.to_datetime(sd[sleep_day_col])
        sd["date"] = sd[sleep_day_col].dt.date
        display_cols.append("date")
    if tot_sleep_col:
        sd["minutes_asleep"] = sd[tot_sleep_col]
        display_cols.append("minutes_asleep")
    if tot_bed_col:
        sd["minutes_in_bed"] = sd[tot_bed_col]
        display_cols.append("minutes_in_bed")
    if display_cols:
        st.dataframe(sd[display_cols].head(20))
        st.markdown("Average sleep (minutes): **{:.1f}**".format(sd["minutes_asleep"].mean() if "minutes_asleep" in sd.columns else float('nan')))
    else:
        st.info("Sleep day file found but columns could not be auto-detected. Showing raw preview.")
        st.dataframe(sd.head(10))
else:
    st.info("No daily sleep file found in kaggle_data or data_raw.")

# ------------- Sleep timeline (minute) ----------------
st.header("Sleep — Minute Timeline")
if minute_sleep_file and minute_sleep_file.exists():
    ms = pd.read_csv(minute_sleep_file)
    # detect minute column formats
    # attempt to find minute and value columns
    minute_col = next((c for c in ms.columns if "minute" in c.lower()), None)
    value_col = next((c for c in ms.columns if c.lower() in ("value","sleepstate","status")), None)
    date_col = next((c for c in ms.columns if "date" in c.lower()), None)
    if minute_col and (value_col or "value" in ms.columns):
        # create timestamp column
        if date_col:
            ms["timestamp"] = pd.to_datetime(ms[date_col].astype(str) + " " + ms[minute_col].astype(str), errors="coerce")
        else:
            # sometimes minute already contains full datetime
            ms["timestamp"] = pd.to_datetime(ms[minute_col], errors="coerce")
        ms = ms.dropna(subset=["timestamp"])
        # asleep flag: value==1 typical
        valcol = value_col if value_col else "value"
        ms["asleep_flag"] = (ms[valcol] == 1) | (ms[valcol].astype(str).str.lower().isin(["asleep","sleep"]))
        # show timeline per day using grouped intervals
        ms["date"] = ms["timestamp"].dt.date
        days = ms["date"].unique()
        fig, ax = plt.subplots(figsize=(10, max(2, len(days)*0.4)))
        y = 0
        yticks = []
        ylabels = []
        for d in sorted(days):
            day_df = ms[ms["date"]==d]
            asleep_intervals = []
            current = None
            for t, asleep in zip(day_df["timestamp"], day_df["asleep_flag"]):
                if asleep:
                    if current is None:
                        current = t
                    end = t
                else:
                    if current is not None:
                        asleep_intervals.append((current, end+pd.Timedelta(minutes=1)))
                        current = None
            if current is not None:
                asleep_intervals.append((current, end+pd.Timedelta(minutes=1)))
            # convert to matplotlib format: (start_in_dt, duration_in_dt)
            intervals = [(pd.Timestamp(s), (pd.Timestamp(e)-pd.Timestamp(s)).total_seconds()/60) for s,e in asleep_intervals]
            # If no intervals, create tiny invisible bar
            if intervals:
                for s, dur in intervals:
                    ax.broken_barh([(s.to_pydatetime().timestamp(), dur)], (y, 0.8), facecolors='tab:blue')
            yticks.append(y+0.4)
            ylabels.append(str(d))
            y += 1
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Time")
        ax.set_title("Sleep Intervals per Day (minuteSleep file)")
        # format x-axis as dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30)
        st.pyplot(fig)
        # create asleep_flag in cleaned df by merging minute flags (fast)
        ms_small = ms.set_index("timestamp")[["asleep_flag"]]
        # resample ms_small to 1-min and align to cleaned df
        ms_res = ms_small.resample("1min").max().fillna(False)
        merged_with_sleep = df.copy()
        merged_with_sleep["timestamp"] = pd.to_datetime(merged_with_sleep["timestamp"], errors="coerce")
        merged_with_sleep = merged_with_sleep.set_index("timestamp")
        # join
        merged_with_sleep = merged_with_sleep.join(ms_res.rename(columns={"asleep_flag":"asleep_flag"}), how="left")
        merged_with_sleep["asleep_flag"] = merged_with_sleep["asleep_flag"].fillna(False)
        merged_with_sleep = merged_with_sleep.reset_index()
        st.success("Merged cleaned data with minute-level asleep_flag (preview below).")
        st.dataframe(merged_with_sleep.head(80))
        st.download_button("Download cleaned_with_sleep.csv", data=merged_with_sleep.to_csv(index=False).encode('utf-8'),
                           file_name="cleaned_with_sleep.csv", mime="text/csv")
    else:
        st.info("Found minute-sleep file but columns didn't match expected patterns. Previewing raw file.")
        st.dataframe(ms.head(20))
else:
    st.info("No minute-level sleep file found; cannot show sleep timeline or asleep_flag. If you want asleep_flag we can merge using daily sleep times (less precise).")

# ------------- Daily Activity ----------------
st.header("Daily Activity (dailyActivity_merged.csv)")
if daily_activity_file:
    da = pd.read_csv(daily_activity_file, parse_dates=[c for c in ["ActivityDate","Date"] if c in open(daily_activity_file).read(2000)])
    # try common columns
    cal_col = next((c for c in da.columns if "calorie" in c.lower()), None)
    dist_col = next((c for c in da.columns if "distance" in c.lower()), None)
    date_col = next((c for c in da.columns if "date" in c.lower()), None)
    if date_col:
        da[date_col] = pd.to_datetime(da[date_col])
        st.dataframe(da[[date_col, cal_col or da.columns[1], dist_col or da.columns[2]]].head(20))
        # aggregated chart
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.bar(da[date_col].dt.date, da[cal_col] if cal_col else 0)
        ax2.set_title("Calories per Day")
        st.pyplot(fig2)
    else:
        st.dataframe(da.head(10))
else:
    st.info("No daily activity file found in kaggle_data.")

# ------------- End -------------
st.markdown("---")
st.write("Files used:")
st.write("cleaned:", CLEANED)
st.write("sleep day:", sleep_day_file if sleep_day_file else "None")
st.write("minute sleep:", minute_sleep_file if minute_sleep_file else "None")
st.write("daily activity:", daily_activity_file if daily_activity_file else "None")
