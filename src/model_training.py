import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
import os

try:
    from src.feature_engineering import FeatureEngineering
except ImportError as e:
    print(f"Import Error: {e}. Ensure 'src' is a package with __init__.py and run from project root.")
    raise

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.fe = FeatureEngineering()

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """Evaluate model performance."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        return metrics

    def train_and_tune(self, X_train, y_train, X_test, y_test, model_name):
        """Train and tune a model with GridSearchCV or RandomizedSearchCV."""
        model = self.models[model_name]
        param_grid = {}
        
        if model_name == 'LogisticRegression':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
            search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='f1', n_jobs=-1, random_state=42)
        elif model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='f1', n_jobs=-1, random_state=42)
        
        with mlflow.start_run(run_name=model_name):
            # Train the model
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Create input example for model signature
            input_example = X_train.head(1).to_dict(orient='records')[0]
            
            # Log parameters, metrics, and model to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"{model_name}_model",
                input_example=input_example,
                registered_model_name=model_name
            )
            
            return best_model, metrics, search.best_params_

    def train_all_models(self, X, y):
        """Train and evaluate all models, select the best based on F1 score."""
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        best_f1 = 0
        
        for model_name in self.models:
            print(f"Training {model_name}...")
            model, metrics, params = self.train_and_tune(X_train, y_train, X_test, y_test, model_name)
            print(f"{model_name} Metrics: {metrics}")
            print(f"{model_name} Best Params: {params}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model = model
                self.best_model_name = model_name
        
        print(f"Best Model: {self.best_model_name} with F1 Score: {best_f1}")
        return self.best_model, self.best_model_name

if __name__ == "__main__":
    try:
        # Set MLflow tracking URI to local server
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Load and preprocess data
        data_path = os.path.join('data', 'processed', 'task2_eda_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        data = pd.read_csv(data_path)
        print("Model Training - Initial DataFrame columns:", data.columns.tolist())
        print("Model Training - Initial DataFrame Index:", data.index.names)
        trainer = ModelTrainer()
        X, y, is_high_risk = trainer.fe.fit_transform(data)
        print("Model Training - Transformed DataFrame columns:", X.columns.tolist())
        print("Model Training - Transformed DataFrame Index:", X.index.names)
        
        # Train models using is_high_risk as target
        best_model, best_model_name = trainer.train_all_models(X, is_high_risk)
        
        # Save the best model
        with mlflow.start_run(run_name=f"best_{best_model_name}"):
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"best_model_{best_model_name}",
                input_example=X.head(1).to_dict(orient='records')[0],
                registered_model_name=f"best_{best_model_name}"
            )
        print(f"Best model {best_model_name} registered and saved.")
    except Exception as e:
        print(f"Error: {e}")