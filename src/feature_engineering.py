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

    def validate_customer_id(self, df):
        """Validate CustomerId for type consistency and nulls."""
        print("Validating CustomerId...")
        print(f"CustomerId type: {df['CustomerId'].dtype}")
        print(f"CustomerId duplicates: {df['CustomerId'].duplicated().sum()}")
        print(f"CustomerId nulls: {df['CustomerId'].isna().sum()}")
        if df['CustomerId'].isna().any():
            raise ValueError("CustomerId contains null values")
        df['CustomerId'] = df['CustomerId'].astype(str)
        print("CustomerId duplicates are expected for transaction data.")
        return df

    def calculate_rfm(self, df):
        """Calculate RFM metrics per CustomerId."""
        df = df.copy().reset_index(drop=True)
        if 'CustomerId' not in df.columns or 'TransactionStartTime' not in df.columns or 'Amount' not in df.columns:
            raise ValueError("Required columns (CustomerId, TransactionStartTime, Amount) are missing")
        
        print("Calculating RFM - Input columns:", df.columns.tolist())
        print("RFM Input Index:", df.index.names)
        
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        if df['TransactionStartTime'].isna().any():
            print(f"Warning: {df['TransactionStartTime'].isna().sum()} nulls in TransactionStartTime. Filling with default date.")
            df['TransactionStartTime'] = df['TransactionStartTime'].fillna(pd.Timestamp('2023-01-01'))
        
        # Calculate RFM metrics
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (df['TransactionStartTime'].max() + pd.Timedelta(days=1) - x.max()).days,  # Recency
            'CustomerId': 'count',  # Frequency
            'Amount': 'sum'  # Monetary
        })
        # Rename columns before resetting index
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        # Reset index
        rfm = rfm.reset_index(drop=False)
        rfm['CustomerId'] = rfm['CustomerId'].astype(str)
        if rfm['Monetary'].lt(0).any():
            print(f"Warning: {rfm['Monetary'].lt(0).sum()} negative Monetary values detected.")
        print("RFM DataFrame columns:", rfm.columns.tolist())
        print("RFM Index:", rfm.index.names)
        print("RFM CustomerId unique:", rfm['CustomerId'].is_unique)
        return rfm

    def cluster_customers(self, rfm):
        """Apply K-Means clustering to RFM metrics."""
        print("Clustering RFM - Input columns:", rfm.columns.tolist())
        print("RFM Cluster Input Index:", rfm.index.names)
        # Scale RFM features
        self.rfm_scaler = StandardScaler()
        rfm_scaled = self.rfm_scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Apply K-Means with 3 clusters
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm['Cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Identify high-risk cluster
        cluster_means = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean().reset_index()
        print("Cluster means:\n", cluster_means)
        
        # Try to find cluster with both min Frequency and Monetary
        matching_clusters = cluster_means[
            (cluster_means['Frequency'] == cluster_means['Frequency'].min()) &
            (cluster_means['Monetary'] == cluster_means['Monetary'].min())
        ]
        if not matching_clusters.empty:
            high_risk_cluster = matching_clusters['Cluster'].iloc[0]
        else:
            # Fallback: select cluster with lowest Monetary
            print("Warning: No cluster has both min Frequency and Monetary. Using lowest Monetary.")
            high_risk_cluster = cluster_means.loc[cluster_means['Monetary'].idxmin(), 'Cluster']
        
        # Assign is_high_risk label
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        rfm_labels = rfm[['CustomerId', 'is_high_risk']].copy().reset_index(drop=True)
        rfm_labels['CustomerId'] = rfm_labels['CustomerId'].astype(str)
        print("RFM Labels columns:", rfm_labels.columns.tolist())
        print("RFM Labels Index:", rfm_labels.index.names)
        print("RFM Labels CustomerId unique:", rfm_labels['CustomerId'].is_unique)
        return rfm_labels

    def extract_time_features(self, df):
        """Extract time-based features from TransactionStartTime if not already present."""
        df = df.copy().reset_index(drop=True)
        if 'TransactionStartTime' not in df.columns:
            raise ValueError("TransactionStartTime column is missing")
        print("Extracting time features - Input columns:", df.columns.tolist())
        print("Time Features Input Index:", df.index.names)
        # Check if time features already exist
        time_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
        if all(col in df.columns for col in time_features):
            print("Time features already present, skipping extraction.")
            return df
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour.fillna(0).astype(int)
        df['TransactionDay'] = df['TransactionStartTime'].dt.day.fillna(1).astype(int)
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month.fillna(1).astype(int)
        df['TransactionYear'] = df['TransactionStartTime'].dt.year.fillna(2023).astype(int)
        print("Time features extracted - Output columns:", df.columns.tolist())
        print("Time Features Output Index:", df.index.names)
        return df

    def create_aggregate_features(self, df):
        """Create aggregate features per customer."""
        df = df.copy().reset_index(drop=True)
        if 'CustomerId' not in df.columns or 'Amount' not in df.columns:
            raise ValueError("CustomerId or Amount column is missing")
        print("Creating aggregate features - Input columns:", df.columns.tolist())
        print("Aggregate Input Index:", df.index.names)
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        # Flatten MultiIndex and rename columns
        agg_features.columns = ['CustomerId', 'TotalTransactionAmount',
                               'AverageTransactionAmount', 'TransactionCount',
                               'StdTransactionAmount']
        agg_features['CustomerId'] = agg_features['CustomerId'].astype(str)
        print("Aggregate features columns:", agg_features.columns.tolist())
        print("Aggregate Features Index:", agg_features.index.names)
        print("Aggregate CustomerId unique:", agg_features['CustomerId'].is_unique)
        # Merge without suffixes
        df = df.merge(agg_features, on='CustomerId', how='left')
        df['StdTransactionAmount'] = df['StdTransactionAmount'].fillna(0)
        print("After aggregate merge - Output columns:", df.columns.tolist())
        print("After Aggregate Index:", df.index.names)
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
        df = df.copy().reset_index(drop=True)
        if target_col not in df.columns:
            raise ValueError(f"{target_col} column is missing")
        print("Starting fit_transform - Input columns:", df.columns.tolist())
        print("Fit Transform Input Index:", df.index.names)
        df = self.validate_customer_id(df)
        df = self.extract_time_features(df)
        df = self.create_aggregate_features(df)
        
        # Calculate RFM and assign is_high_risk
        rfm = self.calculate_rfm(df)
        rfm_labels = self.cluster_customers(rfm)
        # Map is_high_risk using Series
        print("Creating customer risk mapping...")
        rfm_labels = rfm_labels.set_index('CustomerId')['is_high_risk']
        df['is_high_risk'] = df['CustomerId'].map(rfm_labels).fillna(0).astype(int)
        print("After RFM mapping - Output columns:", df.columns.tolist())
        print("After RFM Mapping Index:", df.index.names)
        # Check for duplicates
        if df.columns.duplicated().sum() > 0:
            raise ValueError(f"Duplicate columns detected after RFM mapping: {df.columns[df.columns.duplicated()].tolist()}")
        
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
        print("Final Transformed Index:", transformed_df.index.names)
        return transformed_df, y, df['is_high_risk']

    def transform(self, df):
        """Transform new data using the fitted pipeline, including is_high_risk."""
        df = df.copy().reset_index(drop=True)
        print("Starting transform - Input columns:", df.columns.tolist())
        print("Transform Input Index:", df.index.names)
        df = self.validate_customer_id(df)
        df = self.extract_time_features(df)
        df = self.create_aggregate_features(df)
        
        # Calculate RFM and assign is_high_risk using fitted K-Means
        rfm = self.calculate_rfm(df)
        if self.rfm_scaler and self.kmeans:
            rfm_scaled = self.rfm_scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
            rfm['Cluster'] = self.kmeans.predict(rfm_scaled)
            # Use same logic as fit
            cluster_means = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean().reset_index()
            matching_clusters = cluster_means[
                (cluster_means['Frequency'] == cluster_means['Frequency'].min()) &
                (cluster_means['Monetary'] == cluster_means['Monetary'].min())
            ]
            if not matching_clusters.empty:
                high_risk_cluster = matching_clusters['Cluster'].iloc[0]
            else:
                print("Warning: No cluster has both min Frequency and Monetary (transform). Using lowest Monetary.")
                high_risk_cluster = cluster_means.loc[cluster_means['Monetary'].idxmin(), 'Cluster']
            rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
            rfm_labels = rfm[['CustomerId', 'is_high_risk']].copy().reset_index(drop=True)
            rfm_labels['CustomerId'] = rfm_labels['CustomerId'].astype(str)
            print("RFM labels columns (transform):", rfm_labels.columns.tolist())
            print("RFM Labels Transform Index:", rfm_labels.index.names)
            print("RFM Labels Transform CustomerId unique:", rfm_labels['CustomerId'].is_unique)
            # Map is_high_risk using Series
            print("Creating customer risk mapping (transform)...")
            rfm_labels = rfm_labels.set_index('CustomerId')['is_high_risk']
            df['is_high_risk'] = df['CustomerId'].map(rfm_labels).fillna(0).astype(int)
        else:
            df['is_high_risk'] = 0  # Default if not fitted
        print("After RFM mapping (transform) - Output columns:", df.columns.tolist())
        print("After RFM Mapping Transform Index:", df.index.names)
        # Check for duplicates
        if df.columns.duplicated().sum() > 0:
            raise ValueError(f"Duplicate columns detected after RFM mapping (transform): {df.columns[df.columns.duplicated()].tolist()}")
        
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan if col in self.numerical_cols else 'missing'
        X = df[self.numerical_cols + self.categorical_cols]
        transformed_data = self.pipeline.transform(X)
        feature_names = self.pipeline.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
        transformed_df['is_high_risk'] = df['is_high_risk']
        print("Final transformed DataFrame columns (transform):", transformed_df.columns.tolist())
        print("Final Transformed Transform Index:", transformed_df.index.names)
        return transformed_df, df['is_high_risk']

if __name__ == "__main__":
    try:
        data = pd.read_csv('data/processed/task2_eda_data.csv')
        print("Initial DataFrame columns:", data.columns.tolist())
        print("Initial DataFrame Index:", data.index.names)
        fe = FeatureEngineering()
        transformed_data, y, is_high_risk = fe.fit_transform(data)
        print(transformed_data.head())
        print(f"High-risk customers: {is_high_risk.sum()}")
    except Exception as e:
        print(f"Error: {e}")