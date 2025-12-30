# ===============================
# Imports
# ===============================
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import timedelta

# ===============================
# Resolve project root
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="AI Sales Forecasting",
    layout="wide"
)

st.title("📊 AI Sales Forecasting Dashboard")
st.caption("Upload past sales data and predict future sales using AI")

# ===============================
# Detect Streamlit Cloud
# ===============================
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false") == "true"

# ===============================
# Load Best Model
# ===============================
@st.cache_resource
def load_model():
    model_path = os.path.join(
        BASE_DIR, "models", "best_sales_forecast_model.pkl"
    )
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.header("⚙️ Forecast Settings")

forecast_days = st.sidebar.slider(
    "Days to forecast", min_value=7, max_value=60, value=30
)

st.sidebar.header("📤 Upload Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (date, units_sold)",
    type=["csv"]
)

# ===============================
# Load Data (User Upload > Local > Demo)
# ===============================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])

        if not {"date", "units_sold"}.issubset(df.columns):
            st.error("CSV must contain 'date' and 'units_sold' columns.")
            st.stop()

        df = df.sort_values("date")
        st.success("✅ User data uploaded successfully")
        st.dataframe(df.tail())

    except Exception as e:
        st.error("Invalid CSV file.")
        st.exception(e)
        st.stop()

else:
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed_sales_data.csv")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        st.info("Using locally processed data.")
    else:
        st.warning("⚠️ Processed data not found. Using demo data (cloud mode).")

        dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
        df = pd.DataFrame({
            "date": dates,
            "units_sold": [400 + (i * 3) % 100 for i in range(90)],
        })

# ===============================
# Feature Engineering
# ===============================
df = df.sort_values("date")

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day_of_week"] = df["date"].dt.weekday
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["lag_1"] = df["units_sold"].shift(1)
df["lag_7"] = df["units_sold"].shift(7)
df["rolling_7"] = df["units_sold"].rolling(7).mean()
df["rolling_14"] = df["units_sold"].rolling(14).mean()

df = df.dropna()

# ===============================
# Forecast Logic
# ===============================
last_row = df.iloc[-1:].copy()
future_rows = []
current_date = last_row["date"].values[0]

for _ in range(forecast_days):
    next_date = pd.to_datetime(current_date) + timedelta(days=1)

    features = {
        "day": next_date.day,
        "month": next_date.month,
        "year": next_date.year,
        "day_of_week": next_date.weekday(),
        "is_weekend": int(next_date.weekday() >= 5),
        "lag_1": last_row["units_sold"].values[0],
        "lag_7": last_row["lag_7"].values[0],
        "rolling_7": last_row["rolling_7"].values[0],
        "rolling_14": last_row["rolling_14"].values[0],
    }

    prediction = model.predict(pd.DataFrame([features]))[0]

    future_rows.append({
        "date": next_date,
        "predicted_units_sold": int(prediction)
    })

    last_row["units_sold"] = prediction
    current_date = next_date

forecast_df = pd.DataFrame(future_rows)

# ===============================
# Forecast Output
# ===============================
st.subheader("📅 Forecasted Sales")
st.dataframe(forecast_df, use_container_width=True)

st.download_button(
    "⬇️ Download Forecast CSV",
    forecast_df.to_csv(index=False),
    file_name="future_sales_forecast.csv",
    mime="text/csv"
)

# ===============================
# Plot
# ===============================
st.subheader("📈 Sales Trend")

plt.figure(figsize=(10, 4))
plt.plot(df["date"].tail(60), df["units_sold"].tail(60), label="Historical Sales")
plt.plot(
    forecast_df["date"],
    forecast_df["predicted_units_sold"],
    linestyle="--",
    label="Forecast"
)
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
st.pyplot(plt)

# ===============================
# Model Performance (Local Only)
# ===============================
st.subheader("📊 Model Performance")

if IS_CLOUD:
    st.info("ℹ️ MLflow metrics are available only in local environment.")
else:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(
            f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
        )
        client = MlflowClient()

        experiment = client.get_experiment_by_name("AI_Sales_Forecasting")

        if experiment is None:
            st.warning("No MLflow experiment found.")
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.RMSE ASC"],
                max_results=5
            )

            if not runs:
                st.warning("No MLflow runs available.")
            else:
                metrics_df = pd.DataFrame([
                    {
                        "Run ID": r.info.run_id[:8],
                        "RMSE": r.data.metrics.get("RMSE"),
                        "MAE": r.data.metrics.get("MAE"),
                        "n_estimators": r.data.params.get("n_estimators"),
                    }
                    for r in runs
                ])

                st.dataframe(metrics_df, use_container_width=True)

    except Exception as e:
        st.error("Failed to load MLflow metrics.")
        st.exception(e)
