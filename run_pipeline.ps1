# ==========================================
# AI Sales Forecasting - Automated Pipeline
# ==========================================

$PROJECT_DIR = "C:\Users\krvpv\OneDrive\Desktop\AI_Sales_Forecasting"
$VENV_ACTIVATE = "$PROJECT_DIR\venv\Scripts\activate.ps1"

cd $PROJECT_DIR

# Activate virtual environment
& $VENV_ACTIVATE

Write-Host "Generating sales data..."
python src\data_generator.py

Write-Host "Preprocessing data..."
python src\data_preprocessing.py

Write-Host "Training model with MLflow..."
python src\train_model.py

Write-Host "Pipeline completed successfully"
