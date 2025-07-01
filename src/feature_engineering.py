import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
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
        self.kmeans = None
        self.rfm_scaler = None

    def calculate_rfm(self, df):
        """Calculate RFM metrics per CustomerId."""
        df = df.copy()
        if 'CustomerId' not in df.columns or 'TransactionStartTime' not in df.columns or 'Amount' not in df.columns:
            raise ValueError("Required columns (CustomerId, TransactionStartTime, Amount) are missing")
        
        print("Calculating RFM - Input columns:", df.columns.tolist())
        
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        
        # Set snapshot date as the latest transaction time
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
            'CustomerId': 'count',  # Frequency
            'Amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        print("RFM DataFrame columns:", rfm.columns.tolist())
        return rfm

    def cluster_customers(self, rfm):
        """Apply K-Means clustering to RFM metrics."""
        print("Clustering RFM - Input columns:", rfm.columns.tolist())
        # Scale RFM features
        self.rfm_scaler = StandardScaler()
        rfm_scaled = self.rfm_scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Apply K-Means with 3 clusters
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm['Cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Identify high-risk cluster (lowest Frequency and Monetary)
        cluster_means = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean().reset_index()
        print("Cluster means:\n", cluster_means)
        high_risk_cluster = cluster_means[
            (cluster_means['Frequency'] == cluster_means['Frequency'].min()) &
            (cluster_means['Monetary'] == cluster_means['Monetary'].min())
        ]['Cluster'].iloc[0] if not cluster_means.empty else 0
        
        # Assign is_high_risk label
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        return rfm[['CustomerId', 'is_high_risk']]

    def extract_time_features(self, df):
        """Extract time-based features from TransactionStartTime."""
        df = df.copy()
        if 'TransactionStartTime' not in df.columns:
            raise ValueError("TransactionStartTime column is missing")
        print("Extracting time features - Input columns:", df.columns.tolist())
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour.fillna(0).astype(int)
        df['TransactionDay'] = df['TransactionStartTime'].dt.day.fillna(1).astype(int)
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month.fillna(1).astype(int)
        df['TransactionYear'] = df['TransactionStartTime'].dt.year.fillna(2023).astype(int)
        print("Time features extracted - Output columns:", df.columns.tolist())
        return df

    def create_aggregate_features(self, df):
        """Create aggregate features per customer."""
        if 'CustomerId' not in df.columns or 'Amount' not in df.columns:
            raise ValueError("CustomerId or Amount column is missing")
        print("Creating aggregate features - Input columns:", df.columns.tolist())
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        # Flatten MultiIndex and rename columns
        agg_features.columns = ['CustomerId', 'TotalTransactionAmount',
                               'AverageTransactionAmount', 'TransactionCount',
                               'StdTransactionAmount']
        print("Aggregate features columns:", agg_features.columns.tolist())
        # Merge with suffixes to handle potential duplicates
        df = df.merge(agg_features, on='CustomerId', how='left', suffixes=('', '_agg'))
        # Drop any redundant columns
        df = df.loc[:, ~df.columns.str.endswith('_agg')]
        df['StdTransactionAmount'] = df['StdTransactionAmount'].fillna(0)
        print("After aggregate merge - Output columns:", df.columns.tolist())
        # Check for duplicates
        if df.columns.duplicated().sum() > 0:
            raise ValueError(f"Duplicate columns detected after aggregate merge: {df.columns[df.columns.duplicated()].tolist()}")
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
        """Fit and transform the data using the pipeline, including is_high_risk."""
        if target_col not in df.columns:
            raise ValueError(f"{target_col} column is missing")
        print("Starting fit_transform - Input columns:", df.columns.tolist())
        df = self.extract_time_features(df)
        df = self.create_aggregate_features(df)
        
        # Calculate RFM and assign is_high_risk
        rfm = self.calculate_rfm(df)
        rfm_labels = self.cluster_customers(rfm)
        print("RFM labels columns:", rfm_labels.columns.tolist())
        # Merge is_high_risk, ensuring no duplicate CustomerId
        df = df.merge(rfm_labels[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
        print("After RFM merge - Output columns:", df.columns.tolist())
        # Check for duplicates
        if df.columns.duplicated().sum() > 0:
            raise ValueError(f"Duplicate columns detected after RFM merge: {df.columns[df.columns.duplicated()].tolist()}")
        
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan if col in self.numerical_cols else 'missing'
        X = df[self.numerical_cols + self.categorical_cols]
        y = df[target_col]
        self.build_pipeline()
        transformed_data = self.pipeline.fit_transform(X, y)
        feature_names = self.pipeline.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
        transformed_df['is_high_risk'] = df['is_high_risk']
        print("Final transformed DataFrame columns:", transformed_df.columns.tolist())
        return transformed_df, y, df['is_high_risk']

    def transform(self, df):
        """Transform new data using the fitted pipeline, including is_high_risk."""
        print("Starting transform - Input columns:", df.columns.tolist())
        df = self.extract_time_features(df)
        df = self.create_aggregate_features(df)
        
        # Calculate RFM and assign is_high_risk using fitted K-Means
        rfm = self.calculate_rfm(df)
        if self.rfm_scaler and self.kmeans:
            rfm_scaled = self.rfm_scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
            rfm['Cluster'] = self.kmeans.predict(rfm_scaled)
            high_risk_cluster = self.kmeans.cluster_centers_.argmin(axis=0)[1]  # Lowest Frequency
            rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
            rfm_labels = rfm[['CustomerId', 'is_high_risk']]
            print("RFM labels columns (transform):", rfm_labels.columns.tolist())
            df = df.merge(rfm_labels[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
            df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
        else:
            df['is_high_risk'] = 0  # Default if not fitted
        print("After RFM merge (transform) - Output columns:", df.columns.tolist())
        # Check for duplicates
        if df.columns.duplicated().sum() > 0:
            raise ValueError(f"Duplicate columns detected after RFM merge (transform): {df.columns[df.columns.duplicated()].tolist()}")
        
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan if col in self.numerical_cols else 'missing'
        X = df[self.numerical_cols + self.categorical_cols]
        transformed_data = self.pipeline.transform(X)
        feature_names = self.pipeline.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
        transformed_df['is_high_risk'] = df['is_high_risk']
        print("Final transformed DataFrame columns (transform):", transformed_df.columns.tolist())
        return transformed_df, df['is_high_risk']

if __name__ == "__main__":
    try:
        data = pd.read_csv('data/raw/data.csv')
        print("Initial DataFrame columns:", data.columns.tolist())
        fe = FeatureEngineering()
        transformed_data, y, is_high_risk = fe.fit_transform(data)
        print(transformed_data.head())
        print(f"High-risk customers: {is_high_risk.sum()}")
    except Exception as e:
        print(f"Error: {e}")