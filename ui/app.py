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
# Streamlit page config
# ===============================
st.set_page_config(
    page_title="AI Sales Forecasting",
    layout="wide"
)

st.title("📊 AI Sales Forecasting Dashboard")
st.caption("End-to-End ML Pipeline with Automated Model Selection & MLflow")

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
# Load Processed Data (Cloud-safe)
# ===============================
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_sales_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
else:
    # Cloud fallback demo data
    st.warning("⚠️ Processed data not found. Using demo data (cloud mode).")

    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    df = pd.DataFrame({
        "date": dates,
        "units_sold": [120 + i % 20 for i in range(90)],
        "lag_7": [115 + i % 15 for i in range(90)],
        "rolling_7": [120 + i % 10 for i in range(90)],
        "rolling_14": [120 + i % 8 for i in range(90)],
    })

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.header("⚙️ Forecast Settings")
forecast_days = st.sidebar.slider(
    "Days to forecast", min_value=7, max_value=60, value=30
)

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

    pred = model.predict(pd.DataFrame([features]))[0]

    future_rows.append({
        "date": next_date,
        "predicted_units_sold": int(pred)
    })

    last_row["units_sold"] = pred
    current_date = next_date

forecast_df = pd.DataFrame(future_rows)

# ===============================
# Forecast Output
# ===============================
st.subheader("📅 Forecasted Sales")
st.dataframe(
    forecast_df,
    use_container_width=True
)

# ===============================
# Plot
# ===============================
st.subheader("📈 Sales Trend")

plt.figure(figsize=(10, 4))
plt.plot(
    df["date"].tail(60),
    df["units_sold"].tail(60),
    label="Historical Sales"
)
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
# Model Performance Section
# ===============================
st.subheader("📊 Model Performance")

if IS_CLOUD:
    st.info("ℹ️ MLflow metrics are available only in the local environment.")
else:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(
            f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
        )
        client = MlflowClient()

        experiment = client.get_experiment_by_name(
            "AI_Sales_Forecasting"
        )

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

                st.dataframe(
                    metrics_df,
                    use_container_width=True
                )

    except Exception as e:
        st.error("Failed to load MLflow metrics.")
        st.exception(e)
