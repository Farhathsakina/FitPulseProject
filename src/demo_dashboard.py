import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(page_title="FitPulse Dashboard", layout="wide")
st.title("❤️ FitPulse — Clean, Preprocessed & Sample Data Viewer")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Heart Rate", "Steps", "Sleep", "Full Dataset Preview"])

DATA_DIR = os.path.expanduser("~/FitPulseProject/data_raw")

# Load sample files
heart_file = f"{DATA_DIR}/sample_heart.csv"
steps_file = f"{DATA_DIR}/sample_steps.csv"
sleep_file = f"{DATA_DIR}/sample_sleep.csv"

heart = pd.read_csv(heart_file)
steps = pd.read_csv(steps_file)
sleep = pd.read_csv(sleep_file)

# Convert timestamp
heart["timestamp"] = pd.to_datetime(heart["timestamp"])
steps["timestamp"] = pd.to_datetime(steps["timestamp"])
sleep["timestamp"] = pd.to_datetime(sleep["timestamp"])

# ----------------------------------------
# CLEANING (Milestone-1 Requirement)
# ----------------------------------------
heart_clean = heart.dropna().reset_index(drop=True)
steps_clean = steps.fillna(0).reset_index(drop=True)
sleep_clean = sleep.fillna(method="ffill").fillna(method="bfill").reset_index(drop=True)

st.success("Milestone-1: Cleaning complete — No missing or null values remain!")

# ----------------------------------------
# ❤️ HEART PAGE
# ----------------------------------------
if page == "Heart Rate":
    st.header("❤️ Heart Rate Data")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(heart_clean["timestamp"], heart_clean["bpm"], color="red")
    ax.set_title("Heart Rate Over Time ❤️")
    ax.set_ylabel("BPM")
    st.pyplot(fig)

    st.subheader("Raw Heart Data")
    st.dataframe(heart_clean)

# ----------------------------------------
# 🦶 STEPS PAGE
# ----------------------------------------
elif page == "Steps":
    st.header("🦶 Steps Data")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_clean["timestamp"], steps_clean["steps"], color="blue")
    ax.set_title("Steps Over Time 🦶")
    ax.set_ylabel("Steps")
    st.pyplot(fig)

    st.subheader("Raw Steps Data")
    st.dataframe(steps_clean)

# ----------------------------------------
# 😴 SLEEP PAGE
# ----------------------------------------
elif page == "Sleep":
    st.header("😴 Sleep Timeline")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sleep_clean["timestamp"], sleep_clean["value"], color="purple")
    ax.set_title("Sleep State Timeline 😴 (1 = asleep, 0 = awake)")
    st.pyplot(fig)

    st.subheader("Raw Sleep Data")
    st.dataframe(sleep_clean)

# ----------------------------------------
# 📊 FULL DATASET PAGE
# ----------------------------------------
elif page == "Full Dataset Preview":
    st.header("📊 Full Preprocessed Dataset (Heart, Steps, Sleep)")

    combined = heart_clean.merge(steps_clean, on="timestamp", how="outer")
    combined = combined.merge(sleep_clean, on="timestamp", how="outer")

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    st.success("Data merged and sorted by timestamp.")
    st.dataframe(combined)
