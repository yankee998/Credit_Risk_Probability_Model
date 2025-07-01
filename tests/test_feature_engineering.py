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
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-01 15:00:00', '2023-01-03 09:00:00'],
        'Amount': [100, 200, 150, 50],
        'Value': [100, 200, 150, 50],
        'ProductCategory': ['airtime', 'data_bundles', 'airtime', 'airtime'],
        'ChannelId': ['Channel1', 'Channel2', 'Channel1', 'Channel1'],
        'PricingStrategy': [1, 2, 1, 1],
        'FraudResult': [0, 0, 1, 0]
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
    assert transformed.columns.duplicated().sum() == 0

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
    assert transformed.columns.duplicated().sum() == 0
    assert sum(col.startswith('CustomerId') for col in transformed.columns) == 1

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
    assert transformed.columns.duplicated().sum() == 0

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
    transformed, y, is_high_risk = fe.fit_transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in transformed.columns for col in fe.numerical_cols)
    assert any(col.startswith('cat__') for col in transformed.columns)
    assert 'is_high_risk' in transformed.columns
    assert len(transformed.columns) >= len(fe.numerical_cols) + len(fe.categorical_cols) + 1
    assert is_high_risk.isin([0, 1]).all()
    assert transformed.columns.duplicated().sum() == 0
    assert sum(col.startswith('CustomerId') for col in transformed.columns) == 1

def test_rfm_calculation(sample_data):
    fe = FeatureEngineering()
    rfm = fe.calculate_rfm(sample_data)
    assert all(col in rfm.columns for col in ['CustomerId', 'Recency', 'Frequency', 'Monetary'])
    assert len(rfm) == 3  # 3 unique CustomerIds
    assert rfm[rfm['CustomerId'] == 'C1']['Frequency'].iloc[0] == 2
    assert rfm[rfm['CustomerId'] == 'C1']['Monetary'].iloc[0] == 300
    assert rfm[rfm['CustomerId'] == 'C2']['Recency'].iloc[0] > rfm[rfm['CustomerId'] == 'C3']['Recency'].iloc[0]
    assert rfm.columns.duplicated().sum() == 0

def test_clustering(sample_data):
    fe = FeatureEngineering()
    rfm = fe.calculate_rfm(sample_data)
    rfm_labels = fe.cluster_customers(rfm)
    assert 'is_high_risk' in rfm_labels.columns
    assert rfm_labels['is_high_risk'].isin([0, 1]).all()
    assert len(rfm_labels) == 3
    high_risk_customers = rfm_labels[rfm_labels['is_high_risk'] == 1]['CustomerId']
    rfm_high_risk = rfm[rfm['CustomerId'].isin(high_risk_customers)]
    assert (rfm_high_risk['Frequency'] <= rfm['Frequency'].min()).all() or (rfm_high_risk['Monetary'] <= rfm['Monetary'].min()).all()
    assert rfm_labels.columns.duplicated().sum() == 0

def test_rfm_merge(sample_data):
    fe = FeatureEngineering()
    df = fe.extract_time_features(sample_data)
    df = fe.create_aggregate_features(df)
    rfm = fe.calculate_rfm(df)
    rfm_labels = fe.cluster_customers(rfm)
    df = df.merge(rfm_labels[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    assert 'is_high_risk' in df.columns
    assert df['is_high_risk'].isin([0, 1, np.nan]).all()
    assert df.columns.duplicated().sum() == 0
    assert sum(col.startswith('CustomerId') for col in df.columns) == 1