from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    Amount: float
    Value: float
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    StdTransactionAmount: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    ProductCategory: str
    ChannelId: int
    PricingStrategy: int

    class Config:
        schema_extra = {
            "example": {
                "Amount": 100.0,
                "Value": 100.0,
                "TotalTransactionAmount": 500.0,
                "AverageTransactionAmount": 125.0,
                "TransactionCount": 4,
                "StdTransactionAmount": 50.0,
                "TransactionHour": 10,
                "TransactionDay": 1,
                "TransactionMonth": 1,
                "TransactionYear": 2023,
                "ProductCategory": "airtime",
                "ChannelId": 0,
                "PricingStrategy": 1
            }
        }

class PredictionResponse(BaseModel):
    risk_probability: float

    class Config:
        schema_extra = {
            "example": {
                "risk_probability": 0.75
            }
        }