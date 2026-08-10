# Statistical Process Improvement & Defect Reduction Analysis

## Project Overview

This project uses statistical analysis and machine learning to identify manufacturing process conditions associated with defective production records.

The goal is to support early defect detection and provide data-driven insights that can help manufacturing teams monitor process measurements and reduce potential defects.

## Business Problem

Manufacturing processes generate hundreds of process measurements for every production record. However, the data contains missing values, constant measurements, and a highly imbalanced defect distribution.

The challenge is to identify defective production records while reducing unnecessary process measurements and maintaining useful defect detection capability.

## Solution

The project follows an end-to-end data science workflow:

1. Data loading and understanding
2. Missing-value analysis
3. Removal of highly incomplete measurements
4. Removal of constant measurements
5. Missing-value imputation
6. Statistical analysis
7. Train-test splitting
8. Feature scaling
9. Feature selection
10. Machine learning model development
11. Class imbalance analysis
12. Model comparison
13. Model deployment using Streamlit

## Dataset

The dataset contains:

- 1,567 manufacturing records
- 590 process measurements initially
- 104 defective records
- 1,463 good records

After data cleaning:

- 446 useful measurements remained
- Missing values were imputed
- 50 measurements were selected for the final model

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### Final Model

The selected model is:

**Logistic Regression using 50 selected measurements**

The model was selected based on its ability to identify defective records rather than relying only on overall accuracy.

## Model Results

| Model | Features | Accuracy | Defect Recall |
|---|---:|---:|---:|
| Logistic Regression | 446 | 83.76% | 19.05% |
| Logistic Regression | 50 | 74.52% | **47.62%** |
| Random Forest | 50 | 93.31% | 0.00% |
| XGBoost | 50 | 91.08% | 9.52% |

The final Logistic Regression model identified 10 out of 21 defective records in the test set.

## Deployment

The trained model was deployed using **Streamlit**.

The application allows users to:

- Upload manufacturing measurement data
- Run the trained machine learning pipeline
- Predict potential defects
- View defect probabilities
- Identify records requiring inspection
- Download prediction results

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Google Colab

## Project Architecture

```text
Manufacturing Data
        ↓
Data Cleaning
        ↓
Missing Value Imputation
        ↓
Statistical Analysis
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Defect Prediction
        ↓
Streamlit Deployment
        ↓
Business Decision Support
