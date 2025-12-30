import pandas as pd
import joblib
import yaml
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# Resolve project root
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# MLflow setup
# ===============================
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
mlflow.set_experiment("AI_Sales_Forecasting")

# ===============================
# Load config
# ===============================
with open(os.path.join(BASE_DIR, "config", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# ===============================
# Load data
# ===============================
df = pd.read_csv(
    os.path.join(BASE_DIR, config["data"]["processed_path"]),
    parse_dates=["date"]
)

FEATURES = [
    "day", "month", "year", "day_of_week", "is_weekend",
    "lag_1", "lag_7", "rolling_7", "rolling_14"
]
TARGET = "units_sold"

X = df[FEATURES]
y = df[TARGET]

# Time-aware split
split_index = int(len(df) * config["training"]["train_split"])
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

best_rmse = float("inf")
best_model = None
best_params = None

# ===============================
# Train multiple candidates
# ===============================
for candidate in config["model"]["candidates"]:
    n_estimators = candidate["n_estimators"]

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=config["model"]["random_state"],
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(model, "model")

        print(f"n_estimators={n_estimators} | MAE={mae:.2f} | RMSE={rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = candidate

# ===============================
# Save best model
# ===============================
best_model_path = os.path.join(BASE_DIR, "models", "best_sales_forecast_model.pkl")
joblib.dump(best_model, best_model_path)

print("\n🏆 BEST MODEL SELECTED")
print(f"Params: {best_params}")
print(f"Best RMSE: {best_rmse:.2f}")
print(f"Saved at: {best_model_path}")
