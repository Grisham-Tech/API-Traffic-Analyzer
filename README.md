# 🌐 Universal SRE Command Center
**[🔴 Try the Live Dashboard Here!](https://your-app-url.streamlit.app)**

An ML-driven web application that allows Site Reliability Engineering (SRE) teams to upload server logs and automatically detect bottlenecks and forecast capacity. 

By generalizing the Machine Learning pipeline, this tool acts as a "plug-and-play" command center for any microservice architecture.

## 🚀 Key Features
* **Bring Your Own Data:** Upload any time-series CSV log. The system dynamically maps to your custom metrics (e.g., CPU, Memory, Latency).
* **AI Bottleneck Detection:** Utilizes an `Isolation Forest` model to scan thousands of intervals and instantly flag multi-variable anomalies.
* **Capacity Forecasting:** Uses `Exponential Smoothing` to project 90-day traffic loads based on daily and weekly seasonality patterns.
* **Live Demo Mode:** Includes a built-in demo dataset demonstrating the platform's capabilities on 6 months of banking API traffic.

## 🛠️ Tech Stack
* **Data Processing:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn (Isolation Forest), Statsmodels (Holt-Winters)
* **Frontend:** Streamlit, Matplotlib

## 🏃 How to Run Locally
1. Clone this repository.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Launch the dashboard:
   `streamlit run app.py`
4. Upload your own CSV log or click "Use Demo Data" to explore.# API-Traffic-Analyzer
An ML-driven SRE tool for predicting API bottlenecks and traffic scaling.
