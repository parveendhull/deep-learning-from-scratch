# Customer Churn Prediction using Artificial Neural Network (ANN)

## Overview

This project predicts whether a customer is likely to leave (churn) or stay with a bank using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

The project covers the complete machine learning workflow including:

- Data preprocessing
- Feature engineering
- One-hot encoding
- Feature scaling
- Neural network development
- Model evaluation
- Performance analysis

---

## Dataset

Dataset: Customer Churn Modelling Dataset

Target Variable:

- Exited
  - 0 = Customer Stays
  - 1 = Customer Churns

Features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns
  - RowNumber
  - CustomerId
  - Surname

- One-Hot Encoding for:
  - Geography
  - Gender

- Feature Scaling using StandardScaler

- Train-Test Split (80-20)

---

## Model Architecture

```python
Input Layer (11 Features)

Dense(11, activation='relu')

Dropout(0.3)

Dense(11, activation='relu')

Dropout(0.3)

Dense(1, activation='sigmoid')
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras

---

## Results

Test Accuracy:

```text
85.95%
```

Classification Report:

```text
Precision (Churn): 0.72
Recall (Churn): 0.53
F1-Score (Churn): 0.61
```

Confusion Matrix:

```text
[[1513   82]
 [ 192  213]]
```

---

## Key Learnings

During this project I learned:

- Data preprocessing for deep learning
- One-hot encoding and feature scaling
- Building ANNs using Keras Sequential API
- Binary classification using sigmoid activation
- Binary cross-entropy loss
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Threshold tuning
- Debugging TensorFlow training issues

---

## Future Improvements

- Hyperparameter tuning
- Early Stopping
- ROC-AUC evaluation
- Class imbalance handling
- Compare ANN with:
  - Logistic Regression
  - Random Forest
  - XGBoost

---

## Project Status

Completed and successfully trained an ANN model achieving approximately 86% test accuracy on the Customer Churn dataset.