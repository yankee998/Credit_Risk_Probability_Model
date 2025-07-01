# 🎉 Credit Risk Probability Model

Welcome to the **Credit Risk Probability Model** project! 🚀 This repository implements a state-of-the-art solution to predict credit risk using the Xente dataset, following the B5W5 challenge requirements. Built with Python, MLflow, FastAPI, and Docker, this project takes you from data exploration to a fully containerized API with CI/CD integration. Dive in and explore the journey from Task 1 to Task 6! 🌟

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/yankee998/Credit_Risk_Probability_Model/ci.yml?branch=main&label=CI%20Status)](https://github.com/yankee998/Credit_Risk_Probability_Model/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3119/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Table of Contents
- [🌟 Project Overview](#-project-overview)
- [📊 Task Breakdown](#-task-breakdown)
  - [Task 1: Data Exploration](#task-1-data-exploration)
  - [Task 2: Data Preprocessing](#task-2-data-preprocessing)
  - [Task 3: Feature Engineering](#task-3-feature-engineering)
  - [Task 4: Proxy Target Variable Engineering](#task-4-proxy-target-variable-engineering)
  - [Task 5: Model Training and Tracking](#task-5-model-training-and-tracking)
  - [Task 6: Model Deployment and Continuous Integration](#task-6-model-deployment-and-continuous-integration)
- [🚀 Getting Started](#-getting-started)
- [🛠️ Usage](#-usage)
- [📦 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)
- [📬 Contact](#-contact)
- [📜 License](#-license)

## 🌟 Project Overview
This project leverages the Xente dataset to build a credit risk prediction model. Starting with exploratory data analysis, it progresses through preprocessing, feature engineering, model training with MLflow tracking, and culminates in a containerized FastAPI API with a CI/CD pipeline. Perfect for data scientists, developers, and enthusiasts looking to learn or deploy machine learning solutions!

- **Dataset**: [Google Drive](https://drive.google.com/drive/folders/1tcSdGtCKnMp1qZL_wVC_yVxMrAm28vcx) | [Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge/data)
- **Tech Stack**: Python 3.11, Pandas, Scikit-learn, MLflow, FastAPI, Uvicorn, Docker, GitHub Actions
- **Status**: Actively developed as of July 01, 2025

## 📊 Task Breakdown

### Task 1: Data Exploration
- **Script**: `src/data_exploration.py`
- **Goal**: Analyze the raw Xente dataset (`task1_raw_data.csv`) to understand distributions, missing values, and correlations.
- **Features**:
  - Visualizations of transaction amounts, fraud rates, and categorical variables.
  - Identification of 91,920 duplicate `CustomerId` values and negative `Amount` entries.
- **Output**: `data/processed/task1_eda_output.csv` with insights.

### Task 2: Data Preprocessing
- **Script**: `src/data_preprocessing.py`
- **Goal**: Clean and prepare the data for modeling.
- **Features**:
  - Handles missing values with imputation (mean for numerical, 'missing' for categorical).
  - Removes duplicates where appropriate, retaining 91,920 `CustomerId` duplicates as transactional data.
  - Outputs `data/processed/task2_eda_data.csv` with precomputed time features.

### Task 3: Feature Engineering
- **Script**: `src/feature_engineering.py`
- **Goal**: Enhance the dataset with engineered features.
- **Features**:
  - **Aggregate Features**: Total transaction amount, average, count, and standard deviation per customer.
  - **Time-Based Features**: Hour, day, month, year from `TransactionStartTime` (skipped if precomputed).
  - **Categorical Encoding**: One-hot encoding for `ProductCategory`, `ChannelId`, `PricingStrategy`.
  - **Normalization**: StandardScaler for numerical features.
  - **WoE Transformation**: Custom `CustomWOETransformer` for categorical variables.
- **Tests**: `tests/test_feature_engineering.py` validates feature extraction and aggregation.

### Task 4: Proxy Target Variable Engineering
- **Script**: `src/feature_engineering.py`
- **Goal**: Create a proxy target (`is_high_risk`) using RFM clustering.
- **Features**:
  - **RFM Metrics**: Recency, Frequency, Monetary per `CustomerId`.
  - **Clustering**: K-Means with 3 clusters, identifying high-risk cluster (lowest `Monetary` fallback).
  - **Fixes**: Handles 91,920 duplicates and 192 negative `Monetary` values with warnings.
- **Tests**: Updated `tests/test_feature_engineering.py` for RFM integration.

### Task 5: Model Training and Tracking
- **Script**: `src/model_training.py`
- **Goal**: Train and track models with MLflow.
- **Features**:
  - **Data Splitting**: 80-20 train-test split, stratified by `is_high_risk`.
  - **Model Selection**: Logistic Regression, Random Forest, Gradient Boosting.
  - **Hyperparameter Tuning**: Grid Search (Logistic) and Randomized Search (others).
  - **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC.
  - **Tracking**: MLflow logs parameters, metrics, and registers the best model (F1-based).
- **Setup**: Requires a local MLflow server at `http://localhost:5000`.
- **Tests**: `tests/test_data_processing.py` for `evaluate_model`.

### Task 6: Model Deployment and Continuous Integration
- **Scripts**: `src/api/main.py`, `src/api/pydantic_models.py`
- **Goal**: Deploy the model as a containerized API with CI/CD.
- **Features**:
  - **API**: FastAPI with `/predict` endpoint, loading the best model from MLflow.
  - **Validation**: Pydantic models for request/response data.
  - **Containerization**: `Dockerfile` and `docker-compose.yml` for a Uvicorn-served app.
  - **CI/CD**: `.github/workflows/ci.yml` runs Flake8 linting and Pytest on push to `main`.
- **Deployment**: Accessible at `http://localhost:8000/predict`.

## 🚀 Getting Started
Ready to explore or contribute? Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yankee998/Credit_Risk_Probability_Model.git
   cd "Credit Risk Probability Model"
   ```

2. **Set Up Environment**:
   - Ensure Python 3.11.9 (or 3.13.3) is installed.
   - Create a virtual environment:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare Data**:
   - Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1tcSdGtCKnMp1qZL_wVC_yVxMrAm28vcx) or [Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge/data).
   - Place `task2_eda_data.csv` in `data/processed/`.

4. **Start MLflow Server** (for Tasks 5 & 6):
   - In a separate terminal:
     ```bash
     .\venv\Scripts\activate
     mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///C:/Users/Skyline/Credit%20Risk%20Probability%20Model/mlruns
     ```
   - Access UI at `http://localhost:5000`.

## 🛠️ Usage
- **Run Data Exploration**:
  ```bash
  python -m src.data_exploration
  ```
- **Run Preprocessing**:
  ```bash
  python -m src.data_preprocessing
  ```
- **Run Feature Engineering**:
  ```bash
  python -m src.feature_engineering
  ```
- **Train Models**:
  ```bash
  python -m src.model_training
  ```
- **Test the Code**:
  ```bash
  pytest tests/ -v
  ```

## 📦 Deployment
Deploy the API with Docker:
1. Build and run the container:
   ```bash
   docker-compose up --build
   ```
2. Test the API (e.g., with `curl`):
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"Amount\": 100.0, \"Value\": 100.0, \"TotalTransactionAmount\": 500.0, \"AverageTransactionAmount\": 125.0, \"TransactionCount\": 4, \"StdTransactionAmount\": 50.0, \"TransactionHour\": 10, \"TransactionDay\": 1, \"TransactionMonth\": 1, \"TransactionYear\": 2023, \"ProductCategory\": \"airtime\", \"ChannelId\": 0, \"PricingStrategy\": 1}"
   ```
   - Expected response: `{"risk_probability": <float>}`

## 🤝 Contributing
Love this project? Want to make it better? Here’s how:
- Fork the repository.
- Create a feature branch (`git checkout -b feature-name`).
- Commit your changes (`git commit -m "Add feature"`).
- Push to the branch (`git push origin feature-name`).
- Open a Pull Request with a clear description.

📢 **Issues or Suggestions?** Open an issue on GitHub!

## 📬 Contact
- **Email**: yaredgenana99@gmail.com
- **Last Updated**: 07:04 PM EAT, July 01, 2025

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---