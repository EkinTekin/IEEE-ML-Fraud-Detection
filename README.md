IEEE-CIS Credit Card Fraud Detection
Overview
This repository contains a robust, end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions. Based on the Kaggle IEEE-CIS Fraud Detection dataset, the project tackles the inherent challenges of highly imbalanced financial data (approx. 3.5% fraud rate) through advanced feature engineering, rigorous cross-validation, and algorithmic penalization.

Key Engineering Highlights
Advanced Feature Engineering: Extracted temporal behavioral patterns by converting raw TransactionDT (seconds) into localized TransactionHour and TransactionDay features, significantly boosting the model's ability to detect anomalous nighttime and weekend activities.

Handling Extreme Imbalance: Instead of using traditional (and often flawed) resampling techniques like SMOTE, the class imbalance was handled natively within the XGBoost algorithm using optimal scale_pos_weight penalization, acting as a high-precision "sniper" for fraudulent transactions.

Robust Validation Strategy: Implemented a 5-Fold Stratified Cross-Validation strategy to ensure the model generalizes perfectly to unseen data without overfitting (Data Leakage prevention).

Algorithmic Selection & Ensemble Testing: Conducted empirical tests comparing a Soft Voting Ensemble (Random Forest + XGBoost + Multi-layer Perceptron) against a standalone Tuned XGBoost model. The analysis proved that for tabular, highly imbalanced data, the optimized XGBoost outperformed the diluted probabilities of the ensemble model.

Model Performance & Metrics
The final XGBoost model achieved near-champion tier performance on the validation folds:

Average AUC-ROC: 0.9572 (Excellent class separation capability)

Average AUPRC: 0.7546 (High precision and recall on the minority fraud class)

Project Structure
1_data_prep.py: Merges transaction and identity tables, handling extreme null values (dropping columns with >80% missing data).

2_feature_engineering.py: Imputes missing numeric values (-999 strategy) and applies LabelEncoder for categorical variables.

3_hyperparameter_tuning.py: Utilizes RandomizedSearchCV to find the optimal tree depth, learning rate, and estimators.

4_model_training.py: The core training script featuring temporal engineering, 5-Fold CV, and model serialization (joblib).

5_ultimate_ensemble.py: Experimental script comparing an MLP/RF/XGB ensemble approach.

feature_importance_v2.png: Visual output demonstrating the high predictive power of the engineered temporal features.

How to Run
Clone the repository and install the required dependencies:

Bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib
Ensure the Kaggle dataset (train_transaction.csv and train_identity.csv) is placed inside a data/ folder.

Run the scripts sequentially:

Bash
python 1_data_prep.py
python 2_feature_engineering.py
python 4_model_training.py
The trained model will be saved in the models/ directory as a .pkl file.

Technologies Used
Python 3.x

XGBoost (Gradient Boosting)

Scikit-Learn (Preprocessing, Validation, MLP, Random Forest)

Pandas & NumPy (Data Manipulation)

Matplotlib (Feature Importance Visualization)
