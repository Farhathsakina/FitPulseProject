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

        # Optional: show counts below (small)
        st.write(f"**Total rows (resampled):** {len(df_resampled)}")
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
# -----------------------------
# Milestone 2 (added below Milestone 1) — compact & demo-friendly
# Features -> Prophet (optional) -> KMeans + DBSCAN (both)
# -----------------------------
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# optional libs (safe checks)
try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_OK = True
except Exception:
    TSFRESH_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

st.markdown("---")
st.header("🔬 Milestone 2 — Feature Extraction, Prophet (opt), KMeans & DBSCAN")

# ---------- Controls ----------
with st.sidebar.expander("Milestone 2 (quick)", expanded=False):
    m_window = st.number_input("Window (minutes)", min_value=5, max_value=180, value=60, step=5)
    m_overlap = st.slider("Window overlap %", 0, 80, value=50)
    m_metric = st.selectbox("Metric", ["heart_rate", "steps", "calories"], index=0)
    m_use_tsf = st.checkbox("Use TSFresh (if installed)", value=False)
    m_k = st.number_input("KMeans k", min_value=2, max_value=8, value=3)
    m_db_eps = st.number_input("DBSCAN eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    m_db_min = st.number_input("DBSCAN min_samples", min_value=2, max_value=30, value=5)
    m_prophet_periods = st.number_input("Prophet periods (mins)", min_value=10, max_value=1440, value=120)
    run_m2_all = st.button("▶️ Run Milestone 2 (both)")

# ---------- tiny helpers ----------
def _load_cleaned():
    if 'df_resampled' in globals():
        return df_resampled.copy()
    try:
        df = pd.read_csv(OUT_CLEAN)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    except Exception:
        st.error("Cleaned data not found. Run Milestone 1 or upload cleaned CSV.")
        return None

def _windows(df, col, wmin, overlap):
    """Create overlapping windows and allow a final partial window so small datasets still produce windows."""
    step = max(1, int(wmin * (1 - overlap / 100)))
    wsize = max(1, int(wmin))
    parts = []
    wid = 0
    i = 0
    while i < len(df):
        w = df.iloc[i:i + wsize].copy()
        if w.empty:
            break
        w['window_id'] = wid
        parts.append(w[['window_id', 'timestamp', col]].rename(columns={col: 'value'}))
        wid += 1
        i += step
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def _agg_simple(df, col, wmin, overlap):
    win = _windows(df, col, wmin, overlap)
    if win.empty: return pd.DataFrame()
    g = win.groupby('window_id')['value'].agg(list)
    rows = []
    for wid, vals in g.items():
        a = np.array(vals, dtype=float)
        rows.append({'window_id': wid, 'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()), 'max': float(a.max())})
    feat = pd.DataFrame(rows).set_index('window_id')
    return feat.loc[:, feat.std() > 0]

def _tsfresh_feats(df, col, wmin, overlap):
    if not TSFRESH_OK: return pd.DataFrame()
    win = _windows(df, col, wmin, overlap)
    if win.empty: return pd.DataFrame()
    params = {"mean": None, "standard_deviation": None, "abs_energy": None, "skewness": None}
    feat = extract_features(win, column_id='window_id', column_sort='timestamp', default_fc_parameters=params, n_jobs=1)
    feat = impute(feat)
    return feat.loc[:, feat.std() > 0]

def _fit_prophet(df, col, periods):
    if not PROPHET_OK: return None, None, None
    d = df[['timestamp', col]].dropna().rename(columns={'timestamp': 'ds', col: 'y'})
    if len(d) < 10:
        st.warning("Need >=10 points for Prophet"); return None, None, None
    m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(d)
    fut = m.make_future_dataframe(periods=periods, freq='min'); fc = m.predict(fut)
    merged = d.merge(fc[['ds','yhat','yhat_lower','yhat_upper']], on='ds', how='left'); merged['residual'] = merged['y'] - merged['yhat']
    return m, fc, merged

def _run_both_clusters(feat, k, eps, min_s):
    X = StandardScaler().fit_transform(feat.values)
    # KMeans
    km = KMeans(n_clusters=int(k), random_state=42, n_init=10).fit(X)
    km_labels = km.labels_; km_report = {'inertia': float(km.inertia_), 'n_clusters': int(len(np.unique(km_labels)))}
    # DBSCAN
    db = DBSCAN(eps=float(eps), min_samples=int(min_s)).fit(X)
    db_labels = db.labels_; db_report = {'eps': float(eps), 'min_samples': int(min_s), 'n_clusters': int(len(np.unique(db_labels)))}
    # PCA once and show both
    comp = PCA(n_components=2, random_state=42).fit_transform(X)
    vis = pd.DataFrame({'pc1': comp[:,0], 'pc2': comp[:,1], 'kmeans': km_labels.astype(str), 'dbscan': db_labels.astype(str)})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("KMeans (PCA view)")
        st.plotly_chart(px.scatter(vis, x='pc1', y='pc2', color='kmeans', title='KMeans PCA'), use_container_width=True)
        st.write("KMeans report:", km_report)
    with c2:
        st.subheader("DBSCAN (PCA view)")
        st.plotly_chart(px.scatter(vis, x='pc1', y='pc2', color='dbscan', title='DBSCAN PCA'), use_container_width=True)
        st.write("DBSCAN report:", db_report)
    feat2 = feat.copy(); feat2['kmeans'] = km_labels; feat2['dbscan'] = db_labels
    st.subheader("Feature matrix with labels (first rows)")
    st.dataframe(feat2.head(8))
    return km_labels, db_labels, km_report, db_report

# ---------- run ----------
if run_m2_all:
    st.info("Running: feature extraction → Prophet (optional) → KMeans + DBSCAN")
    dfc = _load_cleaned()
    if dfc is None: st.stop()

    st.subheader("Data preview (Milestone 2)")
    st.dataframe(dfc.head(30))

    # 1) Feature extraction (TSFresh if chosen & available, else simple aggregates)
    st.subheader("1) Feature extraction")
    if m_use_tsf and TSFRESH_OK:
        st.write("Using TSFresh (small set)...")
        feat = _tsfresh_feats(dfc, m_metric, m_window, m_overlap)
        if feat.empty:
            st.info("TSFresh returned no features — using simple aggregates.")
            feat = _agg_simple(dfc, m_metric, m_window, m_overlap)
    else:
        st.write("Using simple rolling-window aggregates.")
        feat = _agg_simple(dfc, m_metric, m_window, m_overlap)

    if feat.empty:
        st.error("No features produced. Try smaller window or check data length."); st.stop()
    st.dataframe(feat.head(6))

    # 2) Prophet (optional)
    st.subheader("2) Trend modeling (Prophet) — optional")
    if m_metric == 'heart_rate' and PROPHET_OK:
        st.write("Fitting Prophet...")
        m, fc, merged = _fit_prophet(dfc, 'heart_rate', int(m_prophet_periods))
        if merged is not None:
            st.write("Forecast preview:"); st.dataframe(fc[['ds','yhat']].head(6))
            fig = px.line(merged, x='ds', y='y', labels={'y':'actual'}); fig.add_scatter(x=fc['ds'], y=fc['yhat'], mode='lines', name='forecast')
            st.plotly_chart(fig, use_container_width=True)
            anoms = merged[np.abs(merged['residual']) > 3 * merged['residual'].std()]
            if not anoms.empty:
                st.warning(f"Prophet found {len(anoms)} anomalies"); st.dataframe(anoms.head(6))
    else:
        st.info("Prophet skipped (install or choose heart_rate).")

    # 3) Clustering (both)
    st.subheader("3) Clustering — KMeans + DBSCAN")
    km_labels, db_labels, kmr, dbr = _run_both_clusters(feat, m_k, m_db_eps, m_db_min)

    st.success("✅ Milestone 2 finished — features → Prophet → KMeans + DBSCAN")
