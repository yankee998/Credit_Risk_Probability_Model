# 🎉 Credit Risk Probability Model 🎉

Welcome to the **Credit Risk Probability Model** project! 🚀 This repository contains an interim submission for Task 1: Understanding Credit Risk, built with Python 3.13.3 on Windows. Let's dive into the details! 🌟

---

## 📊 Project Overview

This project aims to develop a credit risk model using transactional data from the Xente dataset. The interim submission focuses on exploratory data analysis (EDA) and business understanding per the Basel II Accord requirements. Stay tuned for future model development! 🔍

### 🛠️ Tech Stack
- **Language**: Python 3.13.3
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, fastapi
- **Tools**: VS Code, GitHub Actions, Jupyter Notebook
- **CI/CD**: Automated testing and linting with GitHub Actions

---

## ✨ Features

- 📈 Interactive EDA in `1.0-edai.ipynb` with visualizations
- ✅ Automated CI/CD pipeline with linting and testing
- 📝 Business understanding aligned with Basel II
- 🌐 Deployable with FastAPI (future scope)

---

## 🏆 Badges

[![Python Version](https://img.shields.io/badge/Python-3.13.3-blue.svg)](https://www.python.org/downloads/release/python-3133/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yankee998/Credit_Risk_Probability_Model/ci.yml?branch=main&label=CI%20Pipeline)](https://github.com/yankee998/Credit_Risk_Probability_Model/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)



---

## 📋 Credit Scoring Business Understanding

<details>
<summary>🔍 Click to Expand Business Insights</summary>

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Accord prioritizes accurate risk measurement to set capital requirements, requiring interpretable and well-documented models for regulatory validation. This ensures transparency, enabling regulators to assess risk calculations and ensure compliance. An interpretable model like Logistic Regression with Weight of Evidence (WoE) supports this by providing clear, auditable predictions, aligning with Basel II’s governance standards. 🌐

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
The dataset lacks a direct "default" label, necessitating FraudResult as a proxy to estimate credit risk, as fraud may indicate default likelihood for model training. However, business risks include misclassification if the proxy inaccurately reflects default, potentially leading to poor credit decisions, financial losses, or regulatory issues due to unreliable predictions. ⚠️

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
A simple model like Logistic Regression with WoE offers interpretability and regulatory compliance, facilitating validation under Basel II, but it may lack accuracy on complex data. A complex model like Gradient Boosting provides higher predictive power by modeling non-linear patterns, yet its lack of transparency complicates regulatory approval and increases validation efforts. The trade-off balances compliance and explainability against accuracy, often favoring simpler models unless complex ones are thoroughly documented. ⚖️
</details>

---

## 📂 Directory Structure

| Folder/File            | Description                          |
|-------------------------|--------------------------------------|
| `data/raw/`            | Raw datasets (e.g., `data.csv`)      |
| `data/processed/`      | Processed data outputs               |
| `notebooks/1.0-edai.ipynb` | EDA notebook with visualizations |
| `src/`                 | Source code for data processing      |
| `tests/`               | Unit tests                           |
| `.github/workflows/ci.yml` | CI/CD configuration         |
| `requirements.txt`     | Project dependencies                 |

---

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yankee998/Credit_Risk_Probability_Model.git
   cd Credit Risk Probability Model