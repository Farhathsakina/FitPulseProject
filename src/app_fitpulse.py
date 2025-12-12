# app_fitpulse.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from typing import Optional

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

st.set_page_config(page_title="FitPulse Health — Anomaly Detection", layout="wide")

# ----------------- Paths & setup -----------------
BASE = Path.home() / "FitPulseProject"
RAW = BASE / "data_raw"
PROC = BASE / "data_processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
OUT_CLEAN = PROC / "cleaned_fitness_data.csv"

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
    hr = (72 + 8*np.sin(np.linspace(0,3.5,n)) + rng.normal(0,2,n)).round(4)
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
    bpm_df = bpm_df.dropna().reset_index(drop=True)
    if bpm_df.empty:
        return pd.DataFrame({"timestamp":[], "ecg":[]})
    start = pd.to_datetime(bpm_df['timestamp'].iloc[0])
    end = pd.to_datetime(bpm_df['timestamp'].iloc[-1])
    per_sec = bpm_df.set_index('timestamp')['heart_rate'].resample('1S').mean().reindex(pd.date_range(start, end, freq='1S')).ffill().bfill()
    if len(per_sec) == 0:
        return pd.DataFrame({"timestamp":[], "ecg":[]})
    total_seconds = int(min((end - start).total_seconds(), max_seconds))
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

# -----------------------------
# Simple rolling-window fallback feature extractor
# -----------------------------
def simple_window_features(df: pd.DataFrame, metric_col: str, window_size: int, step: Optional[int] = None) -> pd.DataFrame:
    """
    Produces per-window aggregate features: mean, std, min, max, median, q25, q75, skew.
    Each row corresponds to one window.
    """
    df = df.copy()
    if metric_col not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if step is None:
        step = max(1, window_size // 2)
    rows = []
    for i in range(0, len(df) - window_size + 1, step):
        w = df.iloc[i : i + window_size][metric_col].astype(float).dropna()
        if w.empty:
            continue
        agg = {
            "mean": float(w.mean()),
            "std": float(w.std(ddof=0)) if len(w) > 1 else 0.0,
            "min": float(w.min()),
            "max": float(w.max()),
            "median": float(w.median()),
            "q25": float(w.quantile(0.25)),
            "q75": float(w.quantile(0.75)),
            "skew": float(w.skew()) if len(w)>2 else 0.0,
            "window_start": df['timestamp'].iloc[i],
            "window_end": df['timestamp'].iloc[i+window_size-1]
        }
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ----------------------------
# Milestone 2 helper: summary UI (must be defined before use)
# ----------------------------
def milestone2_summary_ui(report: dict,
                          features_df: pd.DataFrame,
                          prophet_count: int,
                          clustering_methods: list,
                          anomalies_count: int,
                          proc_dir: Path = PROC):
    """Displays summary and writes simple txt + json metadata to PROC."""
    st.markdown("## 🎉 Milestone 2 Summary Report")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows Processed", report.get("original_rows", 0))
    with col2:
        st.metric("Features Extracted", report.get("features_extracted", 0))
    with col3:
        st.metric("Prophet Models", prophet_count)
    with col4:
        st.metric("Clustering Methods", len(clustering_methods))

    st.metric("Anomalies Detected", anomalies_count)

    if report.get("error"):
        st.error("Feature extraction error: " + str(report.get("error")))
    else:
        st.success("Milestone 2 Pipeline Completed Successfully ✔️")

    st.markdown("### 📄 Summary (text)")
    st.code(report.get("summary", ""), language="text")

    # Save simple txt & meta
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    plain_lines = [
        "FITPULSE — MILESTONE 2 SUMMARY",
        f"Generated: {now}",
        "",
        f"Rows processed: {report.get('original_rows', 0)}",
        f"Window size: {report.get('window_size', 0)}",
        f"Windows created: {report.get('feature_windows', 0)}",
        f"Features extracted: {report.get('features_extracted', 0)}",
        f"Prophet models trained: {prophet_count}",
        f"Clustering methods: {', '.join(clustering_methods) if clustering_methods else 'None'}",
        f"Anomalies detected: {anomalies_count}",
        f"Extraction error: {report.get('error', 'None')}",
    ]
    plain_text = "\n".join(plain_lines)
    txt_path = proc_dir / "milestone2_summary.txt"
    with open(txt_path, "w") as f:
        f.write(plain_text)

    meta = {
        "generated": now,
        "rows_processed": report.get('original_rows', 0),
        "window_size": report.get('window_size', 0),
        "windows_created": report.get('feature_windows', 0),
        "features_extracted": report.get('features_extracted', 0),
        "prophet_models_trained": prophet_count,
        "clustering_methods": clustering_methods,
        "anomalies_detected": anomalies_count,
        "error": report.get('error', None)
    }
    with open(proc_dir / "milestone2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    try:
        with open(txt_path, "rb") as f:
            st.download_button("⬇️ Download Summary (TXT)", f, file_name="milestone2_summary.txt")
    except Exception:
        st.write("Summary TXT not available for download.")

    st.markdown("### 🔗 Output files on server")
    for name in ["features_m2.csv", "features_with_clusters.csv", "anomalies_m2.csv"]:
        p = proc_dir / name
        st.write(f"- `{name}` — {'exists' if p.exists() else 'missing'}: `{p}`")

    st.success(f"Summary written to: {txt_path}")

# ----------------- Preprocessing (Milestone 1) -----------------
with main:
    st.header("1️⃣ Data Creation — Raw Dataset")
    st.info("This app will create demo data if no raw CSVs are found.")
    if run_btn:
        start_time = time.time()
        st.header("2️⃣ Preprocessing pipeline — running...")
        p = st.progress(0)

        st.info("📌 Generating sample dataset...")
        df = generate_sample(int(n_rows))
        st.write("Raw sample (first rows):")
        st.dataframe(df.head(10))
        p.progress(10)

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

        st.info("⏳ Normalizing timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        p.progress(30)

        st.info("🧽 Removing duplicates...")
        before = len(df)
        df = df.drop_duplicates(subset=['timestamp'])
        after = len(df)
        st.write(f"Rows before: {before}, after de-dup: {after}")
        p.progress(40)

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

        st.info("🩺 Filling missing values...")
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

        st.info("✅ Standardizing and saving cleaned CSV...")
        df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
        for col in df_resampled.select_dtypes(include=[np.number]).columns:
            df_resampled[col] = df_resampled[col].fillna(0)
        df_resampled.to_csv(OUT_CLEAN, index=False)
        p.progress(100)

        st.success(f"Preprocessing complete — cleaned CSV saved to: {OUT_CLEAN}")
        st.write("Cleaned preview:")
        st.dataframe(df_resampled.head(20))

        # Visuals (heart rate, steps, sleep) kept as before
        hr_df = df_resampled[['timestamp','heart_rate']].dropna().copy()
        if not hr_df.empty:
            hr_df['rolling'] = hr_df['heart_rate'].rolling(window=max(1, int(len(hr_df)/8))).mean()
            window = max(1, int(len(hr_df)/8))
            q_high = hr_df['heart_rate'].rolling(window=window).quantile(0.75).fillna(method='bfill')
            q_low  = hr_df['heart_rate'].rolling(window=window).quantile(0.25).fillna(method='ffill')
            hr_df['anomaly'] = False
            if len(hr_df) > 5:
                hr_std = hr_df['heart_rate'].std(ddof=0)
                hr_df['anomaly'] = (
                    (hr_df['heart_rate'] > hr_df['rolling'] + 2 * hr_std) |
                    (hr_df['heart_rate'] < hr_df['rolling'] - 2 * hr_std)
                )
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=hr_df['timestamp'], y=hr_df['heart_rate'], mode='lines', name='BPM', line=dict(width=1), opacity=0.7))
            fig_hr.add_trace(go.Scatter(x=hr_df['timestamp'], y=hr_df['rolling'], mode='lines', name='Smoothed', line=dict(width=2)))
            fig_hr.add_trace(go.Scatter(x=list(hr_df['timestamp']) + list(hr_df['timestamp'][::-1]),
                                        y=list(q_high) + list(q_low[::-1]),
                                        fill='toself', fillcolor='rgba(200,30,30,0.12)', line=dict(color='rgba(255,255,255,0)'),
                                        hoverinfo="skip", showlegend=True, name='IQR'))
            anom = hr_df[hr_df['anomaly'] == True]
            if not anom.empty:
                fig_hr.add_trace(go.Scatter(x=anom['timestamp'], y=anom['heart_rate'], mode='markers', name='Anomaly',
                                            marker=dict(color='yellow', size=9, symbol='x')))
            fig_hr.update_layout(title="❤️ Heart Rate (resampled + smooth + anomalies)", template='plotly_dark', height=420)
            st.plotly_chart(fig_hr, use_container_width=True)

            st.markdown("**ECG-style preview** (synthesized from BPM for demo)")
            ecg_preview = synth_ecg_from_bpm(df_resampled[['timestamp','heart_rate']].rename(columns={'heart_rate':'heart_rate'}), max_seconds=60)
            if not ecg_preview.empty:
                fig_ecg = px.line(ecg_preview, x='timestamp', y='ecg', template='plotly_dark', labels={'ecg':''})
                fig_ecg.update_layout(height=240)
                st.plotly_chart(fig_ecg, use_container_width=True)

        st.markdown("### 🦶 Steps (resampled)")
        fig_steps = px.bar(df_resampled, x='timestamp', y='steps', template='plotly_dark', labels={'timestamp':'Time','steps':'Steps'})
        fig_steps.update_layout(height=300)
        st.plotly_chart(fig_steps, use_container_width=True)

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
if 'df_resampled' in locals():
    st.dataframe(df_resampled, use_container_width=True, height=500)
else:
    st.info("Cleaned dataset not generated yet. Click 'Run Preprocessing' to generate it.")

# -----------------------------
# 🔎 Summary Insights
# -----------------------------
st.markdown("## 🔎 Summary Insights")
if 'df_resampled' in locals():
    try:
        try:
            minutes_per_row = int(''.join(filter(str.isdigit, str(resample_freq)))) or 1
        except Exception:
            minutes_per_row = 1
        min_hr = int(round(df_resampled['heart_rate'].min())) if not df_resampled['heart_rate'].dropna().empty else 0
        max_hr = int(round(df_resampled['heart_rate'].max())) if not df_resampled['heart_rate'].dropna().empty else 0
        avg_steps = round(float(df_resampled['steps'].mean()) if not df_resampled['steps'].dropna().empty else 0.0, 2)
        total_sleep_minutes = int(df_resampled['sleep_flag'].sum()) * minutes_per_row if 'sleep_flag' in df_resampled.columns else 0
        avg_sleep_hours = round(total_sleep_minutes / 60.0, 2)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Min HR", value=f"{min_hr}")
        with c2:
            st.metric(label="Max HR", value=f"{max_hr}")
        with c3:
            st.metric(label="Avg Steps", value=f"{avg_steps}")
        with c4:
            st.metric(label="Avg Sleep (hrs)", value=f"{avg_sleep_hours}")
        st.write(f"**Total rows (resampled):** {len(df_resampled)}")
        st.write(f"**Cleaned CSV saved at:** `{OUT_CLEAN}`")
    except Exception:
        st.write("Summary not available — preprocessing may not have been run completely.")
else:
    st.info("Run preprocessing to see summary insights here.")

if 'start_time' in locals():
    elapsed = time.time() - start_time
    st.success(f"✨ Preprocessing complete in {elapsed:.2f}s — fully cleaned dataset displayed above.")
else:
    st.write("")

# ============================================================
#                   M I L E S T O N E   2
#     Feature Engineering + Forecasting + Clustering + Summary
# ============================================================
st.markdown("---")
st.header("📘 Milestone 2 — Feature Extraction, Clustering & Summary")

# ----------------------------
# Milestone-2 UI controls (now includes metric selection)
# ----------------------------
colA, colB, colC = st.columns(3)

with colA:
    m2_window = st.number_input("Window size (rows)", 10, 2000, 60, 10)
    overlap_pct = st.slider("Window overlap %", 0, 90, 50)
    metric = st.selectbox("Metric to extract", ["heart_rate", "steps", "calories"], index=0)

with colB:
    use_tsfresh = st.checkbox("Use TSFresh (Recommended)", True)
    run_prophet = st.checkbox("Run Prophet Forecasting (optional)", False)

with colC:
    k_kmeans = st.number_input("KMeans clusters", 1, 10, 3)
    dbscan_eps = st.number_input("DBSCAN ε", 0.1, 5.0, 0.5)
    dbscan_min = st.number_input("DBSCAN min_samples", 1, 50, 5)

run_m2 = st.button("▶️ Run Milestone 2 (both)")

# ============================================================
#                TSFresh Feature Extractor Class (compact)
# ============================================================
class TSFreshFeatureExtractor:
    def __init__(self, feature_complexity="minimal"):
        self.feature_complexity = feature_complexity
        self.feature_matrix = None
        self.feature_names = []
        self.extraction_report = {}

    def extract_features(self, df, data_type, window_size):
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
            report["error"] = f"TSFresh error/import failed: {e}"
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
        step = max(1, window_size // 2)
        for i in range(0, len(df) - window_size + 1, step):
            block = df.iloc[i:i+window_size][["timestamp", data_type]].copy()
            block = block.rename(columns={data_type: "value"})
            block["window_id"] = window_id
            rows.append(block[["timestamp", "value", "window_id"]])
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
        if rep.get("error"):
            msg.append(f"ERROR: {rep['error']}")
        return "\n".join(msg)

# ============================================================
#                    RUN MILESTONE-2
# ============================================================
if run_m2:
    if not OUT_CLEAN.exists():
        st.error("Run Milestone-1 first (cleaned_fitness_data.csv missing).")
        st.stop()

    df_clean = pd.read_csv(OUT_CLEAN)
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

    st.info(f"Loaded cleaned data: {len(df_clean)} rows — extracting metric: {metric}")

    extractor = TSFreshFeatureExtractor()
    features = pd.DataFrame()
    report = {}

    # 1) Try TSFresh if user asked for it
    if use_tsfresh:
        features, report = extractor.extract_features(df_clean, metric, int(m2_window))

    # 2) If TSFresh failed or user disabled it, fallback to simple extractor
    if (not report) or (not report.get("success", False)) or features is None or features.empty:
        if use_tsfresh:
            st.warning("TSFresh not available or produced no features — attempting simple rolling-window extractor.")
        else:
            st.info("Using simple rolling-window extractor (TSFresh disabled).")

        features = simple_window_features(df_clean, metric, int(m2_window))
        report = {
            "data_type": metric,
            "original_rows": len(df_clean),
            "window_size": int(m2_window),
            "feature_windows": len(features),
            "features_extracted": 0 if features.empty else features.shape[1],
            "success": False if features.empty else True,
            "error": None,
            "summary": f"Fallback/simple extractor produced {len(features)} windows and {features.shape[1] if not features.empty else 0} features."
        }

    if features is None or features.empty:
        st.error("No features produced. Try smaller window size, increase overlap, or provide more/longer data.")
        milestone2_summary_ui(report, features, prophet_count=0, clustering_methods=[], anomalies_count=0, proc_dir=PROC)
        st.stop()

    # Ensure numeric features only for clustering
    numeric = features.select_dtypes(include=[np.number]).fillna(0)
    feats_path = PROC / "features_m2.csv"
    features.to_csv(feats_path, index=False)
    st.info(f"Saved features → {feats_path}")

    # 3) Optional Prophet forecasting (attempt if requested and enough windows)
    prophet_count = 0
    if run_prophet:
        try:
            from prophet import Prophet
            numeric_windows = numeric
            if numeric_windows.shape[0] >= 5:
                y = numeric_windows.mean(axis=1).reset_index(drop=True)
                ds = pd.date_range(start=df_clean['timestamp'].min(), periods=len(y), freq='T')[:len(y)]
                df_for_prophet = pd.DataFrame({"ds": ds, "y": y})
                model = Prophet()
                model.fit(df_for_prophet)
                future = model.make_future_dataframe(periods=min(60, int(len(y)*0.2)), freq='T')
                forecast = model.predict(future)
                prophet_count = 1
                st.markdown("#### Prophet forecast (preview)")
                st.dataframe(forecast[['ds','yhat']].head(8))
                fig_fore = px.line(forecast, x='ds', y='yhat', title='Prophet forecast (yhat)', template='plotly_dark')
                st.plotly_chart(fig_fore, use_container_width=True)
            else:
                st.info("Not enough windows to run Prophet forecasting.")
        except Exception as e:
            st.warning("Prophet not available or failed: " + str(e))
            prophet_count = 0

    # 4) Clustering — use defensive KMeans and DBSCAN
    clustering_methods_done = []
    if numeric.shape[0] > 0 and numeric.shape[1] >= 1:
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric)

        # Safe KMeans
        try:
            n_samples = X.shape[0]
            requested_k = int(k_kmeans)
            if n_samples <= 0:
                st.info("KMeans skipped: no numeric windows available.")
            else:
                actual_k = min(requested_k, max(1, n_samples))
                if actual_k != requested_k:
                    st.warning(f"KMeans clusters reduced from {requested_k} → {actual_k} because only {n_samples} windows are available.")
                km = KMeans(n_clusters=actual_k, random_state=42, n_init="auto")
                features['kmeans'] = km.fit_predict(X)
                clustering_methods_done.append('kmeans')
                st.info(f"KMeans completed (k={actual_k})")
        except Exception as e:
            st.warning("KMeans failed: " + str(e))

        # DBSCAN
        try:
            db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min))
            features['dbscan'] = db.fit_predict(X)
            clustering_methods_done.append('dbscan')
            st.info("DBSCAN completed")
        except Exception as e:
            st.warning("DBSCAN failed: " + str(e))
    else:
        st.warning("Not enough numeric features for clustering.")

    clustered_path = PROC / "features_with_clusters.csv"
    features.to_csv(clustered_path, index=False)
    st.info(f"Saved clustered features → {clustered_path}")

    # 5) Anomaly detection (3-sigma on per-row mean)
    anomalies_count = 0
    try:
        if numeric.shape[0] > 0:
            window_mean = numeric.mean(axis=1)
            mu = float(window_mean.mean()) if len(window_mean)>0 else 0.0
            sigma = float(window_mean.std()) if float(window_mean.std())>0 else 1.0
            anomalies_mask = (np.abs(window_mean - mu) > 3 * sigma)
            features['anomaly'] = anomalies_mask
            anomalies_count = int(anomalies_mask.sum())
            anomalies_path = PROC / "anomalies_m2.csv"
            features[features['anomaly']].to_csv(anomalies_path, index=False)
            st.warning(f"Anomalous windows detected: {anomalies_count} (saved → {anomalies_path})")
        else:
            st.info("Skipping anomaly detection — no numeric feature rows.")
    except Exception as e:
        st.warning("Anomaly detection error: " + str(e))

    # 6) PCA visualization (if possible)
    try:
        if numeric.shape[1] >= 2 and numeric.shape[0] >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scaler.transform(numeric))
            features['pc1'] = coords[:,0]
            features['pc2'] = coords[:,1]
            color_col = 'kmeans' if 'kmeans' in features.columns else ('dbscan' if 'dbscan' in features.columns else None)
            fig_pca = px.scatter(features, x='pc1', y='pc2', color=features[color_col].astype(str) if color_col else None,
                                 title="PCA of window features", template='plotly_dark')
            st.plotly_chart(fig_pca, use_container_width=True)
    except Exception:
        pass

    # 7) Final summary
    milestone2_summary_ui(report=report,
                         features_df=features,
                         prophet_count=prophet_count,
                         clustering_methods=clustering_methods_done,
                         anomalies_count=anomalies_count,
                         proc_dir=PROC)

    st.success("Milestone 2 finished.")