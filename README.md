# Customer Churn Prediction Project

## Description
This project aims to predict customer churn for a Fintech company using machine learning models. The goal is to identify customers who are likely to churn and take proactive measures to retain them. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Problem Statement
Customer churn is a critical issue for telecommunication companies as acquiring new customers is often more expensive than retaining existing ones. By predicting which customers are likely to churn, the company can implement targeted retention strategies to reduce churn rates and improve customer loyalty.

## Technology/Model Used
- **Programming Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels
- **Models:** Logistic Regression, Gradient Boosting

## Code Functionality

### 1. Data Loading and Preprocessing
- The data is loaded from an Excel file (`Data.xlsx`) containing customer information, charges, and churn status.
- The data is merged into a single DataFrame for analysis.
- Missing values are handled, and categorical variables are encoded.
- Numerical variables are normalized to ensure better model performance.

### 2. Exploratory Data Analysis (EDA)
- **Statistical Summary:** Provides a summary of the numerical and categorical features in the dataset.
- **Target Variable Distribution:** Visualizes the distribution of the target variable (`Churn`).
- **Correlation Analysis:** Generates a correlation matrix to understand relationships between features and the target variable.
- **Feature-Target Relationship:** Examines the relationship between categorical features and `Churn` using count plots.

### 3. Feature Engineering
- **Charges_per_Tenure:** Calculates the total charges divided by tenure to provide an idea of the cost per month.
- **Monthly_Tenure_Ratio:** Calculates the monthly charges divided by tenure to indicate the consistency of spending over time.

### 4. Model Training and Hyperparameter Tuning
- **Logistic Regression:**
  - Hyperparameter tuning is performed using GridSearchCV to find the best parameters.
  - Best parameters found: `C=0.1`, `penalty='l2'`, `solver='lbfgs'`.
- **Gradient Boosting:**
  - Hyperparameter tuning is performed using GridSearchCV to find the best parameters.
  - Best parameters found: `learning_rate=0.01`, `max_depth=5`, `min_samples_leaf=2`, `min_samples_split=10`, `n_estimators=200`, `subsample=0.9`.

### 5. Model Evaluation
- Models are evaluated using the following metrics:
  - **Accuracy:** Proportion of correct predictions.
  - **Precision:** Proportion of positive predictions that are correct.
  - **Recall:** Proportion of actual positives correctly identified.
  - **F1 Score:** Harmonic mean of precision and recall.
  - **ROC AUC:** Area under the Receiver Operating Characteristic curve.
  - **Mean Squared Error (MSE):** Average squared difference between predicted and actual values.
  - **R²:** Proportion of variance in the dependent variable that is predictable from the independent variables.

### 6. Feature Importance and Visualization
- The importance of each feature is calculated for both models.
- Bar charts are created to visualize the feature importances for Logistic Regression and Gradient Boosting.

## Setup

### Python Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/Diego-Cooper/Customer-Churn-Prediction-Project
   cd Customer-Churn-Prediction-Project
