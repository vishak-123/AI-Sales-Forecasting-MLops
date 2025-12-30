import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_sales_data(
    start_date="2023-01-01",
    end_date="2024-12-31",
    n_stores=3,
    n_products=5
):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    data = []

    for store in range(1, n_stores + 1):
        for product in range(1, n_products + 1):
            base_demand = np.random.randint(20, 60)

            for date in dates:
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_boost = 1.2 if date.weekday() >= 5 else 1.0
                noise = np.random.normal(0, 5)

                units_sold = max(
                    int(base_demand * seasonal_factor * weekend_boost + noise), 0
                )

                price = np.random.randint(100, 500)
                revenue = units_sold * price

                data.append([
                    date,
                    store,
                    product,
                    units_sold,
                    price,
                    revenue,
                    date.day_name(),
                    int(date.weekday() >= 5)
                ])

    columns = [
        "date", "store_id", "product_id",
        "units_sold", "price", "revenue",
        "day_of_week", "is_weekend"
    ]

    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "sales_data.csv")

    df = generate_sales_data()
    df.to_csv(output_path, index=False)

    print(f"Sales data generated at {output_path}")