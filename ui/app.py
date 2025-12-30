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
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")

st.title("📊 AI Sales Forecasting Dashboard")
st.caption("Multi-product forecasting with weekly & monthly aggregation")

# ===============================
# Detect Streamlit Cloud
# ===============================
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false") == "true"

# ===============================
# Load Best Model
# ===============================
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "best_sales_forecast_model.pkl")
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.header("⚙️ Forecast Settings")

forecast_days = st.sidebar.slider("Days to forecast", 7, 90, 30)

aggregation = st.sidebar.selectbox(
    "Aggregation Level",
    ["Daily", "Weekly", "Monthly"]
)

st.sidebar.header("📤 Upload Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (date, product_id, units_sold)",
    type=["csv"]
)

# ===============================
# Load Data
# ===============================
if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to begin forecasting.")
    st.stop()

df = pd.read_csv(uploaded_file, parse_dates=["date"])

required_cols = {"date", "product_id", "units_sold"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain date, product_id, and units_sold columns.")
    st.stop()

df = df.sort_values("date")

# ===============================
# Product Selection
# ===============================
products = df["product_id"].unique()
selected_product = st.sidebar.selectbox(
    "Select Product",
    products
)

df = df[df["product_id"] == selected_product]

# ===============================
# Aggregation
# ===============================
if aggregation == "Weekly":
    df = df.resample("W", on="date").sum().reset_index()
elif aggregation == "Monthly":
    df = df.resample("M", on="date").sum().reset_index()

# ===============================
# Feature Engineering
# ===============================
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

step = 1
if aggregation == "Weekly":
    step = 7
elif aggregation == "Monthly":
    step = 30

for _ in range(forecast_days // step):
    next_date = pd.to_datetime(current_date) + timedelta(days=step)

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
        "product_id": selected_product,
        "predicted_units_sold": int(prediction)
    })

    last_row["units_sold"] = prediction
    current_date = next_date

forecast_df = pd.DataFrame(future_rows)

# ===============================
# Output
# ===============================
st.subheader(f"📅 Forecasted Sales — Product {selected_product}")
st.dataframe(forecast_df, use_container_width=True)

st.download_button(
    "⬇️ Download Forecast CSV",
    forecast_df.to_csv(index=False),
    file_name=f"forecast_{selected_product}.csv",
    mime="text/csv"
)

# ===============================
# Plot
# ===============================
st.subheader("📈 Sales Trend")

plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["units_sold"], label="Historical")
plt.plot(
    forecast_df["date"],
    forecast_df["predicted_units_sold"],
    linestyle="--",
    label="Forecast"
)
plt.legend()
st.pyplot(plt)
