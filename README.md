# 💳 IEEE-CIS Credit Card Fraud Detection
> A High-Precision Machine Learning Pipeline for Highly Imbalanced Financial Data.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview
Detecting fraudulent credit card transactions is challenging due to the extreme imbalance of real-world financial data (only **~3.5%** of transactions are fraudulent). This project utilizes the Kaggle IEEE-CIS dataset to build a robust, production-ready XGBoost model that maximizes precision and recall without relying on artificial oversampling (like SMOTE). 

## ✨ Key Engineering Highlights

* ⏱️ **Temporal Feature Engineering:** Transformed raw `TransactionDT` (seconds) into localized `TransactionHour` and `TransactionDay` features. This successfully captured the behavioral patterns of fraudsters (e.g., late-night spikes).
* ⚖️ **Algorithmic Penalization:** Handled the extreme class imbalance natively by optimizing XGBoost's `scale_pos_weight` parameter, forcing the model to heavily penalize false negatives.
* 🛡️ **Leakage-Proof Validation:** Implemented **5-Fold Stratified Cross-Validation** to ensure the model generalizes effectively to unseen data without overfitting.
* 🧠 **Ensemble vs. Standalone Analysis:** Conducted empirical tests comparing a Soft Voting Ensemble (Random Forest + XGBoost + MLP Neural Network) against a tuned XGBoost model, proving that optimized Gradient Boosting outperforms diluted ensemble probabilities on this specific tabular dataset.

---

## 📊 Model Performance

| Model Architecture | Cross-Validation | AUC-ROC Score | AUPRC Score |
| :--- | :---: | :---: | :---: |
| **Optimized XGBoost (Final)** | 5-Fold Stratified | **0.9572** | **0.7546** |
| Soft Voting Ensemble (XGB+RF+MLP) | 5-Fold Stratified | 0.9426 | 0.6965 |

> **Note on Metrics:** While AUC-ROC shows excellent overall separation, the **0.7546 AUPRC** is the critical success metric here, demonstrating a "sniper-like" precision in catching frauds within a heavily skewed (3%) minority class.

---

## 📂 Project Structure

```text
├── data/
│   ├── train_transaction.csv      # Raw Kaggle data (not included in repo)
│   ├── train_identity.csv         # Raw Kaggle data (not included in repo)
│   └── train_final.csv            # Cleaned & Engineered data
├── models/
│   ├── best_fraud_model_v2.pkl    # Serialized Final XGBoost Model
│   └── scaler.pkl                 # StandardScaler for MLP tests
├── 1_data_prep.py                 # Merging & Null handling (>80% drops)
├── 2_feature_engineering.py       # Imputation (-999 strategy) & Encoding
├── 3_model_training.py            # Feature Eng., Training, and 5-Fold CV
├── 4_ultimate_ensemble.py         # Experimental RF + XGB + MLP Voting System
├── feature_importance_v2.png      # Feature importance graph
└── README.md
