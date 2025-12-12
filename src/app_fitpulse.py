# app_fitpulse.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FitPulse Health — Anomaly Detection", layout="wide")

# ----------------- Paths & setup -----------------
BASE = Path.home() / "FitPulseProject"
RAW = BASE / "data_raw"
PROC = BASE / "data_processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
OUT_CLEAN = PROC / "cleaned_fitness_data.csv"

st.set_page_config(page_title="FitPulse Health — Anomaly Detection", layout="wide")
st.markdown("<h1 style='text-align:center;'>❤️ FitPulse Health — Anomaly Detection from Fitness Devices</h1>", unsafe_allow_html=True)
st.markdown("**Pipeline:** data creation → timestamp normalization → missing handling → outlier fix → resample → QC → clean CSV")

# ----------------- Controls in left column -----------------
left, main = st.columns([1, 3])
with left:
    st.markdown("### ⚙️ Controls")
    n_rows = st.number_input("Sample rows to generate (demo)", min_value=10, max_value=1000, value=10, step=10)
    inject_missing = st.checkbox("Inject random missing values (demo)", value=False)
    missing_frac = st.slider("Missing fraction (if injected)", 0.0, 0.5, 0.10, step=0.05)
    outlier_method = st.selectbox("Outlier correction method", ["clip_iqr", "zscore"], index=0)
    resample_freq = st.selectbox("Resample frequency", ["1min", "5min"], index=0)
    run_btn = st.button("🚀 Run Preprocessing")
    st.markdown("---")
    st.markdown("**Tips:** Use 10 rows for quick demo. Toggle missing injection to show handling.")

# ----------------- Helpers -----------------
def generate_sample(n=10, start_ts="2025-01-01 00:00:00"):
    ts = pd.date_range(start=start_ts, periods=n, freq="1T")
    rng = np.random.default_rng(42)
    hr = (72 + 8*np.sin(np.linspace(0,3.5,n)) + rng.normal(0,2,n)).round(22)
    steps = (rng.poisson(2, n) * (rng.random(n) < 0.3)).astype(int)
    calories = (0.9 + 0.5 * rng.random(n) + steps*0.05).round(2)
    sleep_flag = [1 if (i%30)<10 else 0 for i in range(n)]
    activity = rng.choice(["sedentary","light","fair","very_active"], size=n, p=[0.5,0.3,0.15,0.05])
    df = pd.DataFrame({
        "timestamp": ts,
        "heart_rate": hr,
        "steps": steps,
        "calories": calories,
        "sleep_flag": sleep_flag,
        "activity_type": activity
    })
    return df

def synth_ecg_from_bpm(bpm_df, max_seconds=60):
    """Return a small high-res ecg-like DataFrame synthesized from minute BPM."""
    bpm_df = bpm_df.dropna().reset_index(drop=True)
    if bpm_df.empty:
        return pd.DataFrame({"timestamp":[], "ecg":[]})
    start = pd.to_datetime(bpm_df['timestamp'].iloc[0])
    end = pd.to_datetime(bpm_df['timestamp'].iloc[-1])
    total_seconds = int(min((end - start).total_seconds(), max_seconds))
    # build per-second series and 250ms highres
    per_sec = bpm_df.set_index('timestamp')['heart_rate'].resample('1S').mean().reindex(pd.date_range(start, end, freq='1S')).ffill().bfill()
    high_idx = pd.date_range(per_sec.index[0], periods=total_seconds*4+1, freq='250ms')
    bpm_high = per_sec.reindex(high_idx, method='ffill').fillna(method='ffill')
    sig = np.zeros(len(high_idx))
    sigma = 0.06
    for i, t in enumerate(high_idx):
        hb = bpm_high.iloc[i]
        if hb <= 0 or np.isnan(hb):
            continue
        ibi = 60.0 / hb
        seconds_since = (t - per_sec.index[0]).total_seconds()
        phase = seconds_since % ibi
        dist = min(phase, ibi - phase)
        sig[i] = np.exp(-0.5 * (dist/sigma)**2)
    if sig.max() != 0:
        sig = sig / sig.max()
    amp = (bpm_high.values / (bpm_high.max() if bpm_high.max() else 1))
    ecg = sig * (0.5 + 0.8 * amp)
    return pd.DataFrame({"timestamp": high_idx, "ecg": ecg})

with main:

    # -------------------- 1️⃣ Data Creation --------------------
    st.header("1️⃣ Data Creation — Raw Dataset")
    st.markdown(
        """
This section displays the **raw dataset** which i will imported from Kaggle.
It is shown here before applying any preprocessing or cleaning steps.
Use the controls on the left (sample rows, missing values, outlier method, etc.) to configure the preprocessing, missing values, outlier method, etc)
"""
    )

    # Show RAW DATA (safe preview) — MUST be OUTSIDE the run_btn block
    try:
        RAW_HR = str(BASE / "kaggle_data" / "heartrate_seconds.csv")
        raw_hr = pd.read_csv(RAW_HR).head(10)
        st.subheader("❤️ Raw Heart Rate (first 10 rows)")
        st.dataframe(raw_hr)
    except Exception:
        st.info("Raw heart rate file not found (preview skipped).")

    # -------------------- 2️⃣ Preprocessing --------------------
    if run_btn:
        # set start time here (ensure variable exists when used later)
        start_time = time.time()
        st.header("2️⃣ Preprocessing pipeline — running...")
        p = st.progress(0)

        # ---- Load raw files (example) ----
        RAW_HR = str(BASE / "kaggle_data" / "heartrate_seconds.csv")
        RAW_ST = str(BASE / "kaggle_data" / "minuteStepsNarrow.csv")
        RAW_SL = str(BASE / "kaggle_data" / "sleepDay_merged.csv")

        # Read safely (preview)
        try:
            raw_hr = pd.read_csv(RAW_HR).head(10)
            st.subheader("💓 Raw Heart Rate (first 10 rows)")
            st.dataframe(raw_hr)
        except Exception:
            st.warning("Heart rate file not found.")

        try:
            raw_steps = pd.read_csv(RAW_ST).head(10)
            st.subheader("👣 Raw Steps (first 10 rows)")
            st.dataframe(raw_steps)
        except Exception:
            st.warning("Steps file not found.")

        try:
            raw_sleep = pd.read_csv(RAW_SL).head(10)
            st.subheader("😴 Raw Sleep (first 10 rows)")
            st.dataframe(raw_sleep)
        except Exception:
            st.warning("Sleep file not found.")

        # Continue pipeline
        start_time = time.time()
        p.progress(0)

        # 1) Generate or load sample
        st.info("📌 Generating sample dataset...")
        df = generate_sample(int(n_rows))
        st.write("Raw sample (first rows):")
        st.dataframe(df.head(10))
        p.progress(10)

        # 2) optional missing injection
        if inject_missing and missing_frac > 0:
            st.warning("⚠️ Injecting missing values for demo...")
            num = int(len(df) * missing_frac)
            cols = ["heart_rate","steps","calories"]
            rng = np.random.default_rng(123)
            for c in cols:
                if num > 0:
                    idx = rng.choice(df.index, size=num, replace=False)
                    df.loc[idx, c] = np.nan
        p.progress(20)

        # 3) Timestamp normalize
        st.info("⏳ Normalizing timestamps to naive UTC-like format...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        p.progress(30)

        # 4) Remove duplicates
        st.info("🧽 Removing duplicates...")
        before = len(df)
        df = df.drop_duplicates(subset=['timestamp'])
        after = len(df)
        st.write(f"Rows before: {before}, after de-dup: {after}")
        p.progress(40)

        # 5) Outlier detect & correct
        st.info("📉 Detecting & correcting outliers...")
        outlier_report = {"iqr_count":0, "z_count":0}
        if 'heart_rate' in df.columns:
            hr = df['heart_rate'].astype(float)
            q1 = hr.quantile(0.25); q3 = hr.quantile(0.75); iqr = q3 - q1
            low = q1 - 1.5 * iqr; high = q3 + 1.5 * iqr
            iqr_mask = (hr < low) | (hr > high)
            iqr_count = int(iqr_mask.sum())
            mean_hr = hr.mean(); std_hr = hr.std(ddof=0) if hr.std(ddof=0)!=0 else 1.0
            z_mask = ( (hr - mean_hr).abs() / std_hr ) > 3
            z_count = int(z_mask.sum())
            outlier_report['iqr_count'] = iqr_count
            outlier_report['z_count'] = z_count

            if outlier_method == "clip_iqr":
                df['heart_rate'] = df['heart_rate'].clip(lower=low, upper=high)
            else:
                median_hr = hr.median()
                df.loc[z_mask, 'heart_rate'] = median_hr
        p.progress(55)

        # 6) Fill missing values
        st.info("🩺 Filling missing values...")
        # -------------------- 3️⃣ Data Quality Assessment --------------------
        st.header("3️⃣ Data Quality Assessment")

        st.subheader("➡️ Missing Value Summary")
        st.write(df.isna().sum())

        st.subheader("➡️ Outlier Summary")
        st.write(outlier_report)

        st.subheader("➡️ Basic Statistics")
        st.write(df.describe())
        if 'heart_rate' in df.columns:
            df['heart_rate'] = df['heart_rate'].interpolate(method='linear').ffill().bfill()
        if 'steps' in df.columns:
            df['steps'] = df['steps'].fillna(0).astype(int)
        if 'calories' in df.columns:
            df['calories'] = df['calories'].fillna(0.0)
        p.progress(70)

        # 7) Resample to user frequency
        st.info(f"🕒 Resampling to {resample_freq}...")
        df_resampled = df.set_index('timestamp').resample(resample_freq).agg({
            'heart_rate':'mean',
            'steps':'sum',
            'calories':'sum',
            'sleep_flag':'mean',
            'activity_type': lambda x: x.mode().iloc[0] if len(x)>0 else np.nan
        }).reset_index()
        if 'sleep_flag' in df_resampled.columns:
            df_resampled['sleep_flag'] = (df_resampled['sleep_flag'] >= 0.5).astype(int)
        p.progress(85)

        # 8) Final checks & save cleaned CSV
        st.info("✅ Standardizing and saving cleaned CSV...")
        df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
        for col in df_resampled.select_dtypes(include=[np.number]).columns:
            df_resampled[col] = df_resampled[col].fillna(0)
        df_resampled.to_csv(OUT_CLEAN, index=False)
        p.progress(100)

        st.success(f"Preprocessing complete — cleaned CSV saved to: {OUT_CLEAN}")
        st.write("Cleaned preview:")
        st.dataframe(df_resampled.head(20))

        # ---------------- Visualizations ----------------
        # Heart rate: plotly line + smoothed + IQR
        hr_df = df_resampled[['timestamp','heart_rate']].dropna().copy()
        if not hr_df.empty:

            hr_df['rolling'] = hr_df['heart_rate'].rolling(window=max(1, int(len(hr_df)/8))).mean()
            window = max(1, int(len(hr_df)/8))

            q_high = hr_df['heart_rate'].rolling(window=window).quantile(0.75).fillna(method='bfill')
            q_low  = hr_df['heart_rate'].rolling(window=window).quantile(0.25).fillna(method='ffill')

            # 🔍 Add anomaly detection (simple threshold-based using rolling + std)
            hr_df['anomaly'] = False
            if len(hr_df) > 5:
                hr_std = hr_df['heart_rate'].std(ddof=0)
                hr_df['anomaly'] = (
                    (hr_df['heart_rate'] > hr_df['rolling'] + 2 * hr_std) |
                    (hr_df['heart_rate'] < hr_df['rolling'] - 2 * hr_std)
                )

            # Plotting
            fig_hr = go.Figure()

            fig_hr.add_trace(go.Scatter(
                x=hr_df['timestamp'], y=hr_df['heart_rate'],
                mode='lines', name='BPM',
                line=dict(color='firebrick', width=1), opacity=0.7
            ))

            fig_hr.add_trace(go.Scatter(
                x=hr_df['timestamp'], y=hr_df['rolling'],
                mode='lines', name='Smoothed',
                line=dict(color='darkred', width=2)
            ))

            fig_hr.add_trace(go.Scatter(
                x=list(hr_df['timestamp']) + list(hr_df['timestamp'][::-1]),
                y=list(q_high) + list(q_low[::-1]),
                fill='toself', fillcolor='rgba(200,30,30,0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='IQR'
            ))

            # ⭐ Add anomaly markers (only if any)
            anom = hr_df[hr_df['anomaly'] == True]
            if not anom.empty:
                fig_hr.add_trace(go.Scatter(
                    x=anom['timestamp'], y=anom['heart_rate'],
                    mode='markers', name='Anomaly',
                    marker=dict(color='yellow', size=9, symbol='x')
                ))

            fig_hr.update_layout(
                title="❤️ Heart Rate (resampled + smooth + anomalies)",
                template='plotly_dark',
                height=420
            )

            st.plotly_chart(fig_hr, use_container_width=True)

            # ECG-style preview synthesized
            st.markdown("**ECG-style preview** (synthesized from BPM for demo)")
            ecg_preview = synth_ecg_from_bpm(
                df_resampled[['timestamp','heart_rate']].rename(columns={'heart_rate':'heart_rate'}),
                max_seconds=60
            )
            if not ecg_preview.empty:
                fig_ecg = px.line(ecg_preview, x='timestamp', y='ecg', template='plotly_dark', labels={'ecg':''})
                fig_ecg.update_layout(height=240)
                st.plotly_chart(fig_ecg, use_container_width=True)

        # Steps chart
        st.markdown("### 🦶 Steps (resampled)")
        fig_steps = px.bar(df_resampled, x='timestamp', y='steps', template='plotly_dark', labels={'timestamp':'Time','steps':'Steps'})
        fig_steps.update_layout(height=300)
        st.plotly_chart(fig_steps, use_container_width=True)

        # Sleep timeline
        if 'sleep_flag' in df_resampled.columns:
            st.markdown("### 😴 Sleep timeline (1 = asleep)")
            sleep_df = df_resampled[['timestamp','sleep_flag']].copy()
            sleep_df['state'] = sleep_df['sleep_flag'].map({0:'Awake',1:'Asleep'})
            fig_sleep = px.scatter(sleep_df, x='timestamp', y=[1]*len(sleep_df), color='state', template='plotly_dark')
            fig_sleep.update_layout(yaxis=dict(showticklabels=False), height=160)
            st.plotly_chart(fig_sleep, use_container_width=True)

# -----------------------------
# 🧾 Show full cleaned dataset inside dashboard (safe, no errors)
# -----------------------------
st.markdown("## 🧾 Full Cleaned Dataset (No Missing Values)")

# If the pipeline created df_resampled, show it; otherwise ask user to run pipeline.
if 'df_resampled' in locals():
    st.dataframe(df_resampled, use_container_width=True, height=500)
else:
    st.info("Cleaned dataset not generated yet. Click 'Run Preprocessing' to generate it.")

# -----------------------------
# 🔎 Summary Insights (girl-style metric cards)
# -----------------------------
st.markdown("## 🔎 Summary Insights")
if 'df_resampled' in locals():
    try:
        # Determine minutes per row from resample_freq (e.g. "1min" -> 1, "5min" -> 5)
        try:
            minutes_per_row = int(''.join(filter(str.isdigit, str(resample_freq)))) or 1
        except Exception:
            minutes_per_row = 1

        # Safe metric calculations
        min_hr = int(round(df_resampled['heart_rate'].min())) if not df_resampled['heart_rate'].dropna().empty else 0
        max_hr = int(round(df_resampled['heart_rate'].max())) if not df_resampled['heart_rate'].dropna().empty else 0
        avg_steps = round(float(df_resampled['steps'].mean()) if not df_resampled['steps'].dropna().empty else 0.0, 2)

        # Avg sleep in hours: sum of sleep_flag * minutes_per_row -> hours
        total_sleep_minutes = int(df_resampled['sleep_flag'].sum()) * minutes_per_row if 'sleep_flag' in df_resampled.columns else 0
        avg_sleep_hours = round(total_sleep_minutes / 60.0, 2)

        # show as 4 large metrics in one row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Min HR", value=f"{min_hr}")
        with c2:
            st.metric(label="Max HR", value=f"{max_hr}")
        with c3:
            st.metric(label="Avg Steps", value=f"{avg_steps}")
        with c4:
            st.metric(label="Avg Sleep (hrs)", value=f"{avg_sleep_hours}")

        # Optional: show counts and file path below (small)
        st.write(f"**Total rows (resampled):** {len(df_resampled)}")
        st.write(f"**Cleaned CSV saved at:** `{OUT_CLEAN}`")
    except Exception:
        st.write("Summary not available — preprocessing may not have been run completely.")
else:
    st.info("Run preprocessing to see summary insights here.")

# -----------------------------
# ⏱ Show elapsed time if available (safe check)
# -----------------------------
if 'start_time' in locals():
    elapsed = time.time() - start_time
    st.success(f"✨ Preprocessing complete in {elapsed:.2f}s — fully cleaned dataset displayed above.")
else:
    # do not error if the pipeline wasn't executed in this session
    st.write("")  # no-op placeholder
# -----------------------------
# End of file
# -----------------------------
# ============================================================
#                   M I L E S T O N E   2
#     Feature Engineering + Forecasting + Clustering + Summary
# ============================================================

import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

st.markdown("---")
st.header("📘 Milestone 2 — Feature Extraction, Clustering & Summary")

# ----------------------------
# Milestone-2 UI controls
# ----------------------------
colA, colB, colC = st.columns(3)

with colA:
    m2_window = st.number_input("Window size (rows)", 30, 2000, 60, 10)
    overlap_pct = st.slider("Window overlap %", 0, 90, 50)

with colB:
    use_tsfresh = st.checkbox("Use TSFresh (Recommended)", True)
    run_prophet = st.checkbox("Run Prophet Forecasting", False)

with colC:
    k_kmeans = st.number_input("KMeans clusters", 1, 10, 3)
    dbscan_eps = st.number_input("DBSCAN ε", 0.1, 5.0, 0.5)
    dbscan_min = st.number_input("DBSCAN min_samples", 1, 50, 5)

run_m2 = st.button("🚀 Run Milestone 2 Pipeline")

# ============================================================
#                TSFresh Feature Extractor Class
# ============================================================
class TSFreshFeatureExtractor:
    def __init__(self, feature_complexity="minimal"):
        self.feature_complexity = feature_complexity
        self.feature_matrix = None
        self.feature_names = []
        self.extraction_report = {}

    def extract_features(self, df, data_type, window_size):
        from datetime import datetime
        start_time = datetime.now()

        report = {
            "data_type": data_type,
            "original_rows": len(df),
            "window_size": window_size,
            "feature_windows": 0,
            "features_extracted": 0,
            "success": False,
            "error": None,
        }

        df_prep = self._prepare_data(df, data_type, window_size)
        if df_prep is None or df_prep.empty:
            report["error"] = "No valid windows created."
            report["summary"] = self._build_report(report)
            return pd.DataFrame(), report

        try:
            from tsfresh import extract_features
            from tsfresh.utilities.dataframe_functions import impute

            fc_params = {"mean": None, "median": None, "standard_deviation": None}

            features = extract_features(
                df_prep,
                column_id="window_id",
                column_sort="timestamp",
                default_fc_parameters=fc_params,
                disable_progressbar=True,
                n_jobs=1
            )
            features = impute(features)
        except Exception as e:
            report["error"] = str(e)
            report["summary"] = self._build_report(report)
            return pd.DataFrame(), report

        features = features.loc[:, features.nunique() > 1]

        self.feature_matrix = features
        report["feature_windows"] = len(features)
        report["features_extracted"] = features.shape[1]
        report["success"] = True
        report["summary"] = self._build_report(report)

        return features, report

    def _prepare_data(self, df, data_type, window_size):
        if data_type not in df.columns:
            return None

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        rows = []
        window_id = 0
        step = window_size // 2

        for i in range(0, len(df) - window_size + 1, step):
            block = df.iloc[i:i+window_size][["timestamp", data_type]].copy()
            block["value"] = block[data_type]
            block["window_id"] = window_id
            block = block[["timestamp", "value", "window_id"]]
            rows.append(block)
            window_id += 1

        if not rows:
            return None

        return pd.concat(rows, ignore_index=True)

    def _build_report(self, rep):
        msg = [
            "MILESTONE-2 SUMMARY",
            f"Rows Processed: {rep['original_rows']}",
            f"Window Size   : {rep['window_size']}",
            f"Windows Created: {rep['feature_windows']}",
            f"Features Extracted: {rep['features_extracted']}",
        ]
        if rep["error"]:
            msg.append(f"ERROR: {rep['error']}")
        return "\n".join(msg)


# ============================================================
#               Milestone-2 Summary UI writer
# ============================================================
def milestone2_summary_ui(report, features_df, prophet_count, clustering_methods, anomalies_count):
    st.markdown("## 📝 Final Summary — Milestone 2")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Processed", report["original_rows"])
    col2.metric("Features Extracted", report["features_extracted"])
    col3.metric("Anomalies Detected", anomalies_count)

    st.code(report["summary"], language="text")

    summary_path = PROC / "milestone2_summary.txt"
    with open(summary_path, "w") as f:
        f.write(report["summary"])

    st.success(f"Summary saved: {summary_path}")

    with open(summary_path, "rb") as f:
        st.download_button("⬇ Download Summary File", f, file_name="milestone2_summary.txt")


# ============================================================
#                    RUN MILESTONE-2
# ============================================================
if run_m2:

    if not OUT_CLEAN.exists():
        st.error("Run Milestone-1 first (cleaned_fitness_data.csv missing).")
        st.stop()

    df_clean = pd.read_csv(OUT_CLEAN)
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

    st.info("Extracting features...")
    extractor = TSFreshFeatureExtractor()
    features, report = extractor.extract_features(df_clean, "heart_rate", m2_window)

    if not report["success"]:
        st.error("Feature extraction failed.")
        st.text(report["summary"])
        milestone2_summary_ui(report, features, 0, [], 0)
        st.stop()

    # Save feature matrix
    features.to_csv(PROC / "features_m2.csv", index=False)

    # -------------------- CLUSTERING --------------------
    numeric = features.select_dtypes(include=[np.number]).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)

    clustering_methods = []

    # KMeans
    try:
        km = KMeans(n_clusters=int(k_kmeans), random_state=42)
        features["kmeans"] = km.fit_predict(X)
        clustering_methods.append("kmeans")
    except:
        pass

    # DBSCAN
    try:
        db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min))
        features["dbscan"] = db.fit_predict(X)
        clustering_methods.append("dbscan")
    except:
        pass

    features.to_csv(PROC / "features_with_clusters.csv", index=False)

    # -------------------- ANOMALY DETECTION --------------------
    mean_vals = numeric.mean(axis=1)
    mu = mean_vals.mean()
    sigma = mean_vals.std()

    anomalies = (abs(mean_vals - mu) > 3 * sigma)
    features["anomaly"] = anomalies

    anomalies_df = features[features["anomaly"]]
    anomalies_df.to_csv(PROC / "anomalies_m2.csv", index=False)

    anomaly_count = anomalies.sum()

    # -------------------- PCA VISUAL --------------------
    if numeric.shape[1] >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        features["pc1"] = coords[:, 0]
        features["pc2"] = coords[:, 1]

        st.markdown("### 📌 PCA Scatter Plot")
        fig = px.scatter(features, x="pc1", y="pc2",
                         color=features["kmeans"].astype(str) if "kmeans" in features else None,
                         template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------- FINAL SUMMARY --------------------
    milestone2_summary_ui(report, features, prophet_count=0,
                          clustering_methods=clustering_methods,
                          anomalies_count=anomaly_count)