"""
Corporate Profit Prediction API
A FastAPI web service that loads a pre-trained machine learning model 
to predict a startup's profit based on its departmental spending and location.
"""

import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================
# 1. MODEL LOADING
# ==========================================
MODEL_PATH = 'best_model.joblib'

try:
    # Load the pre-trained machine learning pipeline into memory upon startup
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please ensure you run the model training script before starting the API.")
    sys.exit(1)


# ==========================================
# 2. FASTAPI INITIALIZATION
# ==========================================
app = FastAPI(
    title="Startup Profit Prediction API",
    description="An API that estimates corporate profit utilizing financial expenditure data.",
    version="1.0.0"
)


# ==========================================
# 3. PYDANTIC SCHEMAS (DATA VALIDATION)
# ==========================================
class StartupData(BaseModel):
    """Schema defining the expected JSON payload for the prediction endpoint."""
    rd_spend: float = Field(..., description="Total Research and Development expenditure")
    administration: float = Field(..., description="Total Administration expenditure")
    marketing_spend: float = Field(..., description="Total Marketing expenditure")
    state: str = Field(..., description="State where the startup is located (e.g., 'California', 'New York')")


# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.post('/predict')
def predict_profit(data: StartupData) -> dict:
    """
    Receives financial data, applies required feature engineering, 
    and returns the model's profit prediction.
    """
    try:
        # Step 1: Convert the validated JSON payload into a pandas DataFrame
        input_df = pd.DataFrame([{
            'R&D Spend': data.rd_spend, 
            'Administration': data.administration,
            'Marketing Spend': data.marketing_spend,
            'State': data.state
        }])

        # Step 2: Apply Feature Engineering (Matching the training pipeline logic exactly)
        # We add a small epsilon (1e-5) to prevent division by zero errors
        input_df["RD_to_Admin_ratio"] = input_df["R&D Spend"] / (input_df["Administration"] + 1e-5)
        input_df["Marketing_to_Admin_ratio"] = input_df["Marketing Spend"] / (input_df["Administration"] + 1e-5)
        input_df["Total_Spend"] = input_df["R&D Spend"] + input_df["Administration"] + input_df["Marketing Spend"]

        # Step 3: Execute the prediction using the loaded model pipeline
        prediction = model.predict(input_df)

        # Step 4: Return the formatted and rounded result
        return {
            'prediction': round(float(prediction[0]), 2)
        }
        
    except Exception as e:
        # Gracefully handle and report any unexpected errors during data processing or prediction
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {str(e)}")