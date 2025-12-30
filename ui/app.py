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
st.caption("Upload ANY sales dataset and predict future sales")

# ===============================
# Load Trained Model
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

st.sidebar.header("📤 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin forecasting.")
    st.stop()

# ===============================
# Read CSV
# ===============================
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("❌ Unable to read CSV file.")
    st.exception(e)
    st.stop()

st.subheader("🔍 Uploaded Data Preview")
st.dataframe(raw_df.head(), use_container_width=True)

# ===============================
# Column Mapping
# ===============================
st.sidebar.header("🧩 Map Your Columns")

columns = raw_df.columns.tolist()

date_col = st.sidebar.selectbox("Select DATE column", columns)
sales_col = st.sidebar.selectbox("Select SALES column (numeric)", columns)
product_col = st.sidebar.selectbox(
    "Select PRODUCT column (optional)",
    ["None"] + columns
)

# ===============================
# Normalize Columns
# ===============================
df = raw_df.rename(columns={
    date_col: "date",
    sales_col: "units_sold"
})

# ---- DATE VALIDATION ----
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().all():
    st.error("❌ Selected DATE column is not a valid date. Please choose a proper date column.")
    st.stop()

# ---- NUMERIC VALIDATION ----
df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
if df["units_sold"].isna().all():
    st.error("❌ Selected SALES column is not numeric. Please choose a numeric sales column.")
    st.stop()

# ---- PRODUCT HANDLING ----
if product_col != "None":
    df = df.rename(columns={product_col: "product_id"})
else:
    df["product_id"] = "ALL_PRODUCTS"

# ---- CLEAN DATA ----
df = df.dropna(subset=["date", "units_sold"])
df = df.sort_values("date")

if len(df) < 20:
    st.error("❌ Not enough valid data after cleaning (minimum ~20 rows required).")
    st.stop()

# ===============================
# Product Selection
# ===============================
products = df["product_id"].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

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

if len(df) < 10:
    st.error("❌ Not enough data after feature engineering.")
    st.stop()

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

iterations = max(1, forecast_days // step)

for _ in range(iterations):
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
st.subheader(f"📅 Forecast — {selected_product}")
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
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
st.pyplot(plt)

# ===============================
# Info
# ===============================
st.subheader("🤖 Model Info")
st.info(
    "The app performs inference using a pre-trained ML model. "
    "Robust validation and cleaning are applied to handle real-world datasets safely."
)
