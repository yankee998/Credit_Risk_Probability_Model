import os
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI(title="Credit Risk Prediction API")

# Load the best model from MLflow registry
try:
    model_name = "best_GradientBoosting"  # Adjust based on your best model from previous training
    model_version = mlflow.search_registered_models(filter_string=f"name='{model_name}'")[-1].latest_versions[-1].version
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint to predict credit risk probability for new customer data."""
    try:
        # Convert request data to DataFrame
        data = {
            'num__Amount': [request.Amount],
            'num__Value': [request.Value],
            'num__TotalTransactionAmount': [request.TotalTransactionAmount],
            'num__AverageTransactionAmount': [request.AverageTransactionAmount],
            'num__TransactionCount': [request.TransactionCount],
            'num__StdTransactionAmount': [request.StdTransactionAmount],
            'num__TransactionHour': [request.TransactionHour],
            'num__TransactionDay': [request.TransactionDay],
            'num__TransactionMonth': [request.TransactionMonth],
            'num__TransactionYear': [request.TransactionYear],
            'cat__ProductCategory_airtime': [1 if request.ProductCategory == 'airtime' else 0],
            'cat__ProductCategory_data_bundles': [1 if request.ProductCategory == 'data_bundles' else 0],
            'cat__ProductCategory_financial_services': [1 if request.ProductCategory == 'financial_services' else 0],
            'cat__ProductCategory_movies': [1 if request.ProductCategory == 'movies' else 0],
            'cat__ProductCategory_other': [1 if request.ProductCategory == 'other' else 0],
            'cat__ProductCategory_ticket': [1 if request.ProductCategory == 'ticket' else 0],
            'cat__ProductCategory_transport': [1 if request.ProductCategory == 'transport' else 0],
            'cat__ProductCategory_tv': [1 if request.ProductCategory == 'tv' else 0],
            'cat__ProductCategory_utility_bill': [1 if request.ProductCategory == 'utility_bill' else 0],
            'cat__ChannelId_0': [1 if request.ChannelId == 0 else 0],
            'cat__ChannelId_1': [1 if request.ChannelId == 1 else 0],
            'cat__ChannelId_2': [1 if request.ChannelId == 2 else 0],
            'cat__ChannelId_3': [1 if request.ChannelId == 3 else 0],
            'cat__PricingStrategy_0': [1 if request.PricingStrategy == 0 else 0],
            'cat__PricingStrategy_1': [1 if request.PricingStrategy == 1 else 0],
            'cat__PricingStrategy_2': [1 if request.PricingStrategy == 2 else 0],
            'cat__PricingStrategy_3': [1 if request.PricingStrategy == 3 else 0],
            'is_high_risk': [0]  # Placeholder, model will override this
        }
        df = pd.DataFrame(data)

        # Predict probability
        probability = model.predict_proba(df.drop(columns=['is_high_risk']))[:, 1][0]

        return PredictionResponse(risk_probability=probability)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")