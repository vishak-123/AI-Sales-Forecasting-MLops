import pandas as pd

def preprocess_sales_data(file_path):
    # Load data
    df = pd.read_csv(file_path, parse_dates=["date"])

    # Basic validation
    df = df.drop_duplicates()
    df = df[df["units_sold"] >= 0]

    # Aggregate daily total sales (across all stores & products)
    daily_sales = (
        df.groupby("date")
        .agg({"units_sold": "sum", "revenue": "sum"})
        .reset_index()
        .sort_values("date")
    )

    # Time-based features
    daily_sales["day"] = daily_sales["date"].dt.day
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["year"] = daily_sales["date"].dt.year
    daily_sales["day_of_week"] = daily_sales["date"].dt.weekday
    daily_sales["is_weekend"] = (daily_sales["day_of_week"] >= 5).astype(int)

    # Lag features
    daily_sales["lag_1"] = daily_sales["units_sold"].shift(1)
    daily_sales["lag_7"] = daily_sales["units_sold"].shift(7)

    # Rolling averages
    daily_sales["rolling_7"] = daily_sales["units_sold"].rolling(7).mean()
    daily_sales["rolling_14"] = daily_sales["units_sold"].rolling(14).mean()

    # Drop rows with NaN (created by lag/rolling)
    daily_sales = daily_sales.dropna().reset_index(drop=True)

    return daily_sales


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(BASE_DIR, "data", "sales_data.csv")
    processed_path = os.path.join(BASE_DIR, "data", "processed_sales_data.csv")

    df_processed = preprocess_sales_data(raw_path)
    df_processed.to_csv(processed_path, index=False)

    print(f"Processed data saved at {processed_path}")