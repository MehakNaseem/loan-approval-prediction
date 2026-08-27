# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or rejected based on applicant details like income, credit history, education, and marital status.

## 📊 Dataset

- **Source:** [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Size:** 614 rows, 13 columns
- **Target variable:** `Loan_Status` (Approved / Rejected)

## 🔧 What This Project Does

1. **Data Cleaning** — Handled missing values in columns like `Gender`, `Married`, `LoanAmount`, and `Credit_History` using mode/median imputation
2. **Encoding** — Converted categorical columns (Gender, Married, Education, Property_Area) into numerical format for the model
3. **Feature Scaling** — Standardized numeric features using `StandardScaler` for better model performance
4. **Model Training** — Trained and compared two classification models:
   - Logistic Regression
   - Random Forest Classifier
5. **Evaluation** — Assessed models using accuracy, precision, recall, F1-score, and confusion matrix

## 📈 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.9% |
| Random Forest | 75.6% |

Logistic Regression performed better on this dataset. Both models showed stronger performance at predicting approvals than rejections, likely due to class imbalance in the dataset (more approved cases than rejected).

## 🛠️ Tech Stack

- Python
- pandas
- scikit-learn

## 🚀 How to Run

```bash
pip install pandas scikit-learn
python loan_prediction.py
```

## 📌 Key Learnings

- Importance of handling missing data before modeling
- Encoding categorical variables for ML compatibility
- Why feature scaling matters for algorithms like Logistic Regression
- Comparing multiple models rather than relying on a single one
- Interpreting evaluation metrics beyond just accuracy (especially recall, for imbalanced classes)