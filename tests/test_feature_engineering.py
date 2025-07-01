import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineering

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C1', 'C2', 'C1'],
        'Amount': [100, 200, 150],
        'Value': [100, 200, 150],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-03 14:00:00'],
        'ProductCategory': ['cat1', 'cat2', 'cat1'],
        'ChannelId': ['ch1', 'ch2', 'ch1'],
        'PricingStrategy': [1, 2, 1],
        'FraudResult': [0, 1, 0],
        'TransactionHour': [10, 12, 14],
        'TransactionDay': [1, 2, 3],
        'TransactionMonth': [1, 1, 1],
        'TransactionYear': [2023, 2023, 2023]
    })
    return data

def test_extract_time_features(sample_data):
    fe = FeatureEngineering()
    df = fe.extract_time_features(sample_data)
    assert 'TransactionHour' in df.columns
    assert 'TransactionDay' in df.columns
    assert 'TransactionMonth' in df.columns
    assert 'TransactionYear' in df.columns
    assert df['TransactionHour'].iloc[0] == 10
    assert df['TransactionDay'].iloc[0] == 1
    assert df['TransactionMonth'].iloc[0] == 1
    assert df['TransactionYear'].iloc[0] == 2023

def test_create_aggregate_features(sample_data):
    fe = FeatureEngineering()
    df = fe.create_aggregate_features(sample_data)
    assert 'TotalTransactionAmount' in df.columns
    assert 'AverageTransactionAmount' in df.columns
    assert 'TransactionCount' in df.columns
    assert 'StdTransactionAmount' in df.columns
    assert df[df['CustomerId'] == 'C1']['TransactionCount'].iloc[0] == 2
    assert df[df['CustomerId'] == 'C1']['TotalTransactionAmount'].iloc[0] == 250

def test_rfm_calculation(sample_data):
    fe = FeatureEngineering()
    rfm = fe.calculate_rfm(sample_data)
    assert 'CustomerId' in rfm.columns
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert len(rfm) == 2  # Two unique CustomerIds
    assert rfm[rfm['CustomerId'] == 'C1']['Frequency'].iloc[0] == 2

def test_rfm_mapping(sample_data):
    fe = FeatureEngineering()
    transformed_data, y, is_high_risk = fe.fit_transform(sample_data)
    assert 'is_high_risk' in transformed_data.columns
    assert is_high_risk.isin([0, 1]).all()
    assert transformed_data.columns.duplicated().sum() == 0
    assert 'CustomerId' not in transformed_data.columns  # CustomerId not in final output