# 📊 AI Sales Forecasting System

An end-to-end AI-powered sales forecasting application that predicts future daily sales using Machine Learning and provides an interactive web dashboard.

---

## 🚀 Project Overview
This project simulates a real-world retail sales environment and builds a complete pipeline:
- Sales data generation
- Data preprocessing & feature engineering
- Machine Learning model training
- Model deployment using Streamlit

---

## 🧠 Tech Stack
- Python 3.11
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

---

## 🏗️ Architecture
1. Generate realistic daily sales data (multi-store, multi-product)
2. Aggregate and preprocess time-series data
3. Train Random Forest regression model
4. Save trained model
5. Deploy model via Streamlit Web UI

---

## 📈 Features
- Forecast future sales (7–60 days)
- Interactive dashboard
- Time-series visualization
- End-to-end ML pipeline

---

## ▶️ How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py
