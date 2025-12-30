import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import mlflow
from datetime import timedelta

IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false") == "true"


# ===============================
# Resolve project root
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")

st.title("📊 AI Sales Forecasting Dashboard")
st.caption("End-to-End ML Pipeline with Automated Model Selection & MLflow")

# ===============================
# Load best model
# ===============================
@st.cache_resource
def load_best_model():
    model_path = os.path.join(BASE_DIR, "models", "best_sales_forecast_model.pkl")
    return joblib.load(model_path)

model = load_best_model()

# ===============================
# Load data (Cloud-safe)
# ===============================
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_sales_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
else:
    st.warning("Processed data not found. Using demo data for cloud deployment.")

    # Create minimal demo data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    df = pd.DataFrame({
        "date": dates,
        "units_sold": [100 + i % 20 for i in range(90)],
        "lag_7": [95 + i % 20 for i in range(90)],
        "rolling_7": [100 + i % 15 for i in range(90)],
        "rolling_14": [100 + i % 10 for i in range(90)],
    })


# ===============================
# Sidebar
# ===============================
st.sidebar.header("⚙️ Forecast Settings")
forecast_days = st.sidebar.slider("Days to forecast", 7, 60, 30)

# ===============================
# Forecast logic
# ===============================
last_row = df.iloc[-1:].copy()
future_rows = []

current_date = last_row["date"].values[0]

for _ in range(forecast_days):
    next_date = pd.to_datetime(current_date) + timedelta(days=1)

    row = {
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

    prediction = model.predict(pd.DataFrame([row]))[0]
    row["predicted_units_sold"] = int(prediction)
    row["date"] = next_date

    future_rows.append(row)
    last_row["units_sold"] = prediction
    current_date = next_date

forecast_df = pd.DataFrame(future_rows)

# ===============================
# Display forecast
# ===============================
st.subheader("📅 Forecasted Sales")
st.dataframe(forecast_df[["date", "predicted_units_sold"]])

# ===============================
# Plot
# ===============================
st.subheader("📈 Sales Trend")

plt.figure(figsize=(10, 4))
plt.plot(df["date"].tail(60), df["units_sold"].tail(60), label="Historical")
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
# MLflow metrics
# ===============================
st.subheader("📊 Model Performance")

if IS_CLOUD:
    st.info("MLflow metrics available in local environment only.")
else:
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("AI_Sales_Forecasting")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.RMSE ASC"],
        max_results=5
    )

    metrics_data = []
    for run in runs:
        metrics_data.append({
            "run_id": run.info.run_id[:8],
            "RMSE": run.data.metrics.get("RMSE"),
            "MAE": run.data.metrics.get("MAE"),
            "n_estimators": run.data.params.get("n_estimators")
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)
