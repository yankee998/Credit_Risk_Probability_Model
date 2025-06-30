# tests/test_feature_engineering.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineering, CustomWOETransformer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-01 15:00:00'],
        'Amount': [100, 200, 150],
        'Value': [100, 200, 150],
        'ProductCategory': ['airtime', 'data_bundles', 'airtime'],
        'ChannelId': ['Channel1', 'Channel2', 'Channel1'],
        'PricingStrategy': [1, 2, 1],
        'FraudResult': [0, 0, 1]
    })

def test_extract_time_features(sample_data):
    fe = FeatureEngineering()
    transformed = fe.extract_time_features(sample_data)
    assert 'TransactionHour' in transformed.columns
    assert 'TransactionDay' in transformed.columns
    assert 'TransactionMonth' in transformed.columns
    assert 'TransactionYear' in transformed.columns
    assert transformed['TransactionHour'].iloc[0] == 10
    assert transformed['TransactionDay'].iloc[0] == 1
    assert transformed['TransactionMonth'].iloc[0] == 1
    assert transformed['TransactionYear'].iloc[0] == 2023

def test_create_aggregate_features(sample_data):
    fe = FeatureEngineering()
    transformed = fe.create_aggregate_features(sample_data)
    assert 'TotalTransactionAmount' in transformed.columns
    assert 'AverageTransactionAmount' in transformed.columns
    assert 'TransactionCount' in transformed.columns
    assert 'StdTransactionAmount' in transformed.columns
    assert transformed['TransactionCount'].iloc[0] == 2
    assert transformed['TotalTransactionAmount'].iloc[0] == 300
    assert transformed['AverageTransactionAmount'].iloc[0] == 150
    assert abs(transformed['StdTransactionAmount'].iloc[0] - 70.710678) < 1e-5

def test_custom_woe_transformer(sample_data):
    woe = CustomWOETransformer()
    X = sample_data[['ProductCategory', 'ChannelId']]
    y = sample_data['FraudResult']
    woe.fit(X, y)
    transformed = woe.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed['ProductCategory'].iloc[0] != 0
    assert transformed['ChannelId'].iloc[1] != 0
    assert len(woe.woe_dict['ProductCategory']) == 2
    assert len(woe.woe_dict['ChannelId']) == 2
    assert woe.get_feature_names_out() == ['ProductCategory', 'ChannelId']

def test_missing_column_handling():
    fe = FeatureEngineering()
    invalid_data = pd.DataFrame({
        'CustomerId': ['C1'],
        'Amount': [100]
    })
    with pytest.raises(ValueError, match="TransactionStartTime column is missing"):
        fe.extract_time_features(invalid_data)

def test_pipeline_output(sample_data):
    fe = FeatureEngineering()
    transformed, y = fe.fit_transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in transformed.columns for col in fe.numerical_cols)
    assert any(col.startswith('cat__') for col in transformed.columns)
    assert len(transformed.columns) >= len(fe.numerical_cols) + len(fe.categorical_cols)