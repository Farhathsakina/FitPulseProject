import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="FitPulse Viewer", layout="wide")
st.title("FitPulse — Cleaned Data Viewer")

CLEAN = Path.home() / "FitPulseProject" / "data_processed" / "cleaned_fitness_data.csv"

if not CLEAN.exists():
    st.error(f"Cleaned file not found: {CLEAN}")
else:
    df = pd.read_csv(CLEAN, parse_dates=["timestamp"])
    st.success(f"Loaded cleaned file ({len(df)} rows).")
    st.dataframe(df.head(100))

    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    axes[0].plot(df["timestamp"], df["bpm"])
    axes[0].set_title("Heart Rate")

    axes[1].bar(df["timestamp"].iloc[:200], df["steps"].iloc[:200], width=0.0006)
    axes[1].set_title("Steps (first 200 mins)")

    st.pyplot(fig)
