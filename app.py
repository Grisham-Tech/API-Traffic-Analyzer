import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Universal SRE Intelligence", layout="wide", page_icon="🌐")

# Hide standard Streamlit branding for a cleaner SaaS look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🌐 Universal SRE Command Center")
st.markdown("Upload any server log CSV to automatically detect bottlenecks and forecast future capacity.")
st.markdown("---")

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
        if 'total_success' in df.columns and 'total_fail' in df.columns:
            df['total_traffic'] = df['total_success'] + df['total_fail']
            df['error_rate'] = df['total_fail'] / df['total_traffic'].replace(0, 1)
    except FileNotFoundError:
        st.sidebar.error("Demo file 'api_traffic_data.csv' not found in folder.")

# --- Main App Logic ---
if df is not None:
    st.sidebar.success("✅ Data loaded successfully!")
    
    st.sidebar.header("2. Map Your Data")
    st.sidebar.markdown("Tell the AI which columns to look at.")
    
    time_col = st.sidebar.selectbox("Timestamp Column", df.columns)
    group_col = st.sidebar.selectbox("Service/Group Column (Optional)", ["None"] + list(df.columns))
    target_col = st.sidebar.selectbox("Traffic Volume Column (For Forecasting)", df.columns, index=len(df.columns)-1)
    anomaly_cols = st.sidebar.multiselect("Metrics for Bottlenecks (e.g., CPU, Memory, Errors)", df.columns)
    
    if st.sidebar.button("🚀 Run ML Analysis", use_container_width=True):
        with st.spinner("Crunching the data..."):
            
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col)
            
            if len(anomaly_cols) > 0:
                df[anomaly_cols] = df[anomaly_cols].ffill().bfill().fillna(0)
            
            # --- KPI METRIC CARDS ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Log Entries", value=f"{len(df):,}")
            with col2:
                total_vol = df[target_col].sum() if pd.api.types.is_numeric_dtype(df[target_col]) else "N/A"
                st.metric(label="Total Volume Processed", value=f"{total_vol:,.0f}" if total_vol != "N/A" else "N/A")
            with col3:
                services_count = df[group_col].nunique() if group_col != "None" else 1
                st.metric(label="Services Monitored", value=services_count)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📊 Traffic Overview", "🚨 Bottleneck Detection", "📈 Capacity Forecast"])
            
            # Force Matplotlib to use a dark theme to match Streamlit
            plt.style.use('dark_background')
            
            # --- TAB 1: OVERVIEW ---
            with tab1:
                st.subheader("Historical Traffic Volume")
                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor('#0E1117') # Matches Streamlit dark background
                ax.set_facecolor('#0E1117')
                
                if group_col != "None":
                    for group in df[group_col].unique():
                        group_data = df[df[group_col] == group].set_index(time_col)
                        daily = group_data[target_col].resample('D').sum()
                        ax.plot(daily.index, daily.values, label=str(group), linewidth=2)
                    ax.legend(frameon=False)
                else:
                    daily = df.set_index(time_col)[target_col].resample('D').sum()
                    ax.plot(daily.index, daily.values, label="Total Traffic", color="#00F0FF", linewidth=2)
                    
                ax.set_ylabel("Total Requests")
                ax.grid(alpha=0.1) # Subtle grid
                st.pyplot(fig)
                
            # --- TAB 2: ANOMALIES ---
            with tab2:
                if len(anomaly_cols) == 0:
                    st.warning("Please select at least one metric for bottleneck detection in the sidebar.")
                else:
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
                    
                    st.error(f"⚠️ **AI Detected {len(anomalies)} critical system bottlenecks.**")
                    display_cols = [time_col] + ([group_col] if group_col != "None" else []) + anomaly_cols
                    st.dataframe(anomalies[display_cols].head(50), use_container_width=True)
            
            # --- TAB 3: FORECASTING ---
            with tab3:
                st.subheader("90-Day Traffic Forecast")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                fig2.patch.set_facecolor('#0E1117')
                ax2.set_facecolor('#0E1117')
                
                try:
                    if group_col != "None":
                        for group in df[group_col].unique():
                            group_data = df[df[group_col] == group].set_index(time_col)
                            daily = group_data[target_col].resample('D').sum()
                            
                            model = ExponentialSmoothing(daily, trend='add', seasonal='add', seasonal_periods=7).fit()
                            forecast = model.forecast(90)
                            
                            p = ax2.plot(daily.index, daily.values, label=f"{group} (Hist)", linewidth=2)
                            ax2.plot(forecast.index, forecast.values, '--', color=p[0].get_color(), alpha=0.6, linewidth=2)
                    else:
                        daily = df.set_index(time_col)[target_col].resample('D').sum()
                        model = ExponentialSmoothing(daily, trend='add', seasonal='add', seasonal_periods=7).fit()
                        forecast = model.forecast(90)
                        
                        ax2.plot(daily.index, daily.values, label="Historical", color="#00F0FF", linewidth=2)
                        ax2.plot(forecast.index, forecast.values, '--', label="Forecast", color="#00F0FF", alpha=0.5, linewidth=2)
                        
                    ax2.set_ylabel("Daily Volume")
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
                    ax2.grid(alpha=0.1)
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning("Not enough continuous historical data to calculate a 90-day seasonal forecast.")
else:
    st.info("👈 Please upload a CSV file or check the 'Use Demo Data' box in the sidebar to begin.")
