import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="Universal SRE Intelligence", layout="wide", page_icon="🌐")
st.title("🌐 Universal SRE Command Center")
st.markdown("Upload any server log CSV to automatically detect bottlenecks and forecast future capacity.")

# --- Sidebar: Data Input ---
st.sidebar.header("1. Data Source")
use_demo = st.sidebar.checkbox("Use Demo Data (API Traffic)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type="csv")

df = None

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_demo:
    try:
        df = pd.read_csv('api_traffic_data.csv')
        # Auto-engineer metrics for the demo data so it works seamlessly
        if 'total_success' in df.columns and 'total_fail' in df.columns:
            df['total_traffic'] = df['total_success'] + df['total_fail']
            df['error_rate'] = df['total_fail'] / df['total_traffic'].replace(0, 1)
    except FileNotFoundError:
        st.sidebar.error("Demo file 'api_traffic_data.csv' not found in folder.")

# --- Main App Logic ---
if df is not None:
    st.sidebar.success("Data loaded successfully!")
    
    st.sidebar.header("2. Map Your Data")
    st.sidebar.markdown("Tell the AI which columns to look at.")
    
    # Dynamic column mapping
    time_col = st.sidebar.selectbox("Timestamp Column", df.columns)
    group_col = st.sidebar.selectbox("Service/Group Column (Optional)", ["None"] + list(df.columns))
    target_col = st.sidebar.selectbox("Traffic Volume Column (For Forecasting)", df.columns, index=len(df.columns)-1)
    anomaly_cols = st.sidebar.multiselect("Metrics for Bottlenecks (e.g., CPU, Memory, Errors)", df.columns)
    
    if st.sidebar.button("🚀 Run ML Analysis"):
        with st.spinner("Running Machine Learning models..."):
            
            # Clean and format datetime
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col)
            
            # Handle missing values for ML
            if len(anomaly_cols) > 0:
                df[anomaly_cols] = df[anomaly_cols].ffill().bfill().fillna(0)
            
            # Create Tabs for the UI
            tab1, tab2, tab3 = st.tabs(["📊 Traffic Overview", "🚨 Bottleneck Detection", "📈 Capacity Forecast"])
            
            # --- TAB 1: OVERVIEW ---
            with tab1:
                st.subheader("Historical Traffic Volume")
                fig, ax = plt.subplots(figsize=(12, 5))
                
                if group_col != "None":
                    for group in df[group_col].unique():
                        group_data = df[df[group_col] == group].set_index(time_col)
                        daily = group_data[target_col].resample('D').sum()
                        ax.plot(daily.index, daily.values, label=str(group))
                    ax.legend()
                else:
                    daily = df.set_index(time_col)[target_col].resample('D').sum()
                    ax.plot(daily.index, daily.values, label="Total Traffic", color="blue")
                    
                ax.set_ylabel("Total Requests")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
            # --- TAB 2: ANOMALIES ---
            with tab2:
                if len(anomaly_cols) == 0:
                    st.warning("Please select at least one metric for bottleneck detection in the sidebar.")
                else:
                    st.subheader("AI Bottleneck Detection (Isolation Forest)")
                    df['is_anomaly'] = 0
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    
                    if group_col != "None":
                        for group in df[group_col].unique():
                            idx = df[group_col] == group
                            preds = iso.fit_predict(df.loc[idx, anomaly_cols])
                            df.loc[idx, 'is_anomaly'] = np.where(preds == -1, 1, 0)
                    else:
                        preds = iso.fit_predict(df[anomaly_cols])
                        df['is_anomaly'] = np.where(preds == -1, 1, 0)
                        
                    anomalies = df[df['is_anomaly'] == 1]
                    st.error(f"**Detected {len(anomalies)} critical system bottlenecks.**")
                    
                    # Display the anomalies cleanly
                    display_cols = [time_col] + ([group_col] if group_col != "None" else []) + anomaly_cols
                    st.dataframe(anomalies[display_cols].head(50))
            
            # --- TAB 3: FORECASTING ---
            with tab3:
                st.subheader("90-Day Traffic Forecast (Exponential Smoothing)")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                
                try:
                    if group_col != "None":
                        for group in df[group_col].unique():
                            group_data = df[df[group_col] == group].set_index(time_col)
                            daily = group_data[target_col].resample('D').sum()
                            
                            # Train & Forecast
                            model = ExponentialSmoothing(daily, trend='add', seasonal='add', seasonal_periods=7).fit()
                            forecast = model.forecast(90)
                            
                            p = ax2.plot(daily.index, daily.values, label=f"{group} (Hist)")
                            ax2.plot(forecast.index, forecast.values, '--', color=p[0].get_color(), alpha=0.7)
                    else:
                        daily = df.set_index(time_col)[target_col].resample('D').sum()
                        model = ExponentialSmoothing(daily, trend='add', seasonal='add', seasonal_periods=7).fit()
                        forecast = model.forecast(90)
                        
                        ax2.plot(daily.index, daily.values, label="Historical", color="blue")
                        ax2.plot(forecast.index, forecast.values, '--', label="Forecast", color="blue", alpha=0.7)
                        
                    ax2.set_ylabel("Daily Volume")
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax2.grid(alpha=0.3)
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning("Not enough continuous historical data to calculate a 90-day seasonal forecast.")
else:
    st.info("👈 Please upload a CSV file or check the 'Use Demo Data' box in the sidebar to begin.")