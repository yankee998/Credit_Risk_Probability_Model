import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime

class CustomWOETransformer:
    """Custom Weight of Evidence (WoE) transformer."""
    def __init__(self):
        self.woe_dict = {}
        self.input_features = None

    def fit(self, X, y):
        """Calculate WoE for each feature."""
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        self.input_features = X.columns.tolist()
        self.woe_dict = {}
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                woe_table = pd.DataFrame({
                    'value': X[col].value_counts().index,
                    'count': X[col].value_counts().values
                })
                woe_table['positive'] = 0
                woe_table['negative'] = 0
                for idx, val in enumerate(woe_table['value']):
                    mask = X[col] == val
                    woe_table.loc[idx, 'positive'] = y[mask].sum()
                    woe_table.loc[idx, 'negative'] = mask.sum() - y[mask].sum()
                woe_table['positive'] = woe_table['positive'].clip(lower=0.5)
                woe_table['negative'] = woe_table['negative'].clip(lower=0.5)
                woe_table['woe'] = np.log((woe_table['positive'] / y.sum()) / 
                                         (woe_table['negative'] / (len(y) - y.sum())))
                self.woe_dict[col] = woe_table.set_index('value')['woe'].to_dict()
        return self

    def transform(self, X):
        """Apply WoE transformation."""
        X = pd.DataFrame(X, columns=self.input_features) if isinstance(X, np.ndarray) else X.copy()
        for col in self.woe_dict:
            if col in X.columns:
                X[col] = X[col].map(self.woe_dict[col]).fillna(0)
        return X

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        return self.input_features if input_features is None else input_features

class FeatureEngineering:
    def __init__(self):
        self.pipeline = None
        self.numerical_cols = ['Amount', 'Value', 'TotalTransactionAmount',
                              'AverageTransactionAmount', 'TransactionCount',
                              'StdTransactionAmount', 'TransactionHour',
                              'TransactionDay', 'TransactionMonth', 'TransactionYear']
        self.categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    def extract_time_features(self, df):
        """Extract time-based features from TransactionStartTime."""
        df = df.copy()
        if 'TransactionStartTime' not in df.columns:
            raise ValueError("TransactionStartTime column is missing")
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour.fillna(0).astype(int)
        df['TransactionDay'] = df['TransactionStartTime'].dt.day.fillna(1).astype(int)
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month.fillna(1).astype(int)
        df['TransactionYear'] = df['TransactionStartTime'].dt.year.fillna(2023).astype(int)
        return df

    def create_aggregate_features(self, df):
        """Create aggregate features per customer."""
        if 'CustomerId' not in df.columns or 'Amount' not in df.columns:
            raise ValueError("CustomerId or Amount column is missing")
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        agg_features.columns = ['CustomerId', 'TotalTransactionAmount',
                               'AverageTransactionAmount', 'TransactionCount',
                               'StdTransactionAmount']
        df = df.merge(agg_features, on='CustomerId', how='left')
        df['StdTransactionAmount'] = df['StdTransactionAmount'].fillna(0)
        return df

    def build_pipeline(self):
        """Build sklearn pipeline for feature engineering."""
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('woe', CustomWOETransformer())
        ])

        self.pipeline = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, self.numerical_cols),
            ('cat', categorical_pipeline, self.categorical_cols)
        ])

    def fit_transform(self, df, target_col='FraudResult'):
        """Fit and transform the data using the pipeline."""
        if target_col not in df.columns:
            raise ValueError(f"{target_col} column is missing")
        df = self.extract_time_features(df)
        df = self.create_aggregate_features(df)
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan if col in self.numerical_cols else 'missing'
        X = df[self.numerical_cols + self.categorical_cols]
        y = df[target_col]
        self.build_pipeline()
        transformed_data = self.pipeline.fit_transform(X, y)
        feature_names = self.pipeline.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
        return transformed_df, y

    def transform(self, df):
        """Transform new data using the fitted pipeline."""
        df = self.extract_time_features(df)
        df = self.create_aggregate_features(df)
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan if col in self.numerical_cols else 'missing'
        X = df[self.numerical_cols + self.categorical_cols]
        transformed_data = self.pipeline.transform(X)
        feature_names = self.pipeline.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
        return transformed_df

if __name__ == "__main__":
    try:
        data = pd.read_csv('data/raw/data.csv')
        fe = FeatureEngineering()
        transformed_data, y = fe.fit_transform(data)
        print(transformed_data.head())
    except Exception as e:
        print(f"Error: {e}")