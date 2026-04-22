# 🏦 Loan Approval Prediction — Machine Learning Project

A complete end-to-end machine learning pipeline that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal attributes. Eight classification algorithms are trained, compared, and evaluated to identify the best-performing model.

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── loan_approval_dataset.csv        # Raw dataset
├── loan_approval_prediction.ipynb   # Main notebook (EDA → Preprocessing → Training → Evaluation)
└── README.md                        # This file
```

---

##  Dataset

**File:** `loan_approval_dataset.csv`  
**Rows:** 4,269 loan applications  
**Columns:** 13 (including target)  
**Missing Values:** None  
**Source:** Structured tabular data with applicant financial profiles

### Target Variable

| Value | Meaning | Count |
|-------|---------|-------|
| `Approved` | Loan was approved | 2,656 (62.2%) |
| `Rejected` | Loan was rejected | 1,613 (37.8%) |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | Integer | Unique identifier — **dropped** during preprocessing |
| `no_of_dependents` | Integer | Number of financial dependents |
| `education` | Categorical | `Graduate` / `Not Graduate` |
| `self_employed` | Categorical | `Yes` / `No` |
| `income_annum` | Integer | Annual income (INR) |
| `loan_amount` | Integer | Requested loan amount (INR) |
| `loan_term` | Integer | Loan repayment term (months) |
| `cibil_score` | Integer | Credit score (300–900); higher = better creditworthiness |
| `residential_assets_value` | Integer | Value of residential property owned |
| `commercial_assets_value` | Integer | Value of commercial property owned |
| `luxury_assets_value` | Integer | Value of luxury assets (vehicles, etc.) |
| `bank_asset_value` | Integer | Value of liquid bank assets |
| `loan_status` | Categorical | **Target** — `Approved` or `Rejected` |

### Key Observations from EDA

- **CIBIL score** is the strongest predictor — approved applicants cluster at scores above 700, while rejected applicants cluster below 500
- **Higher income** correlates strongly with approval, though it is not sufficient alone
- **Graduate** applicants have a higher approval rate than non-graduates
- **Asset values** (especially bank and luxury assets) positively influence approval
- No multicollinearity issues detected in the correlation matrix

---

## ⚙️ Preprocessing Pipeline

The following steps are applied before model training (all handled in Section 4 of the notebook):

| Step | Detail |
|------|--------|
| **Drop ID** | `loan_id` removed — carries no predictive signal |
| **Encode `education`** | `Graduate` → 0, `Not Graduate` → 1 |
| **Encode `self_employed`** | `No` → 0, `Yes` → 1 |
| **Encode target** | `Approved` → 1, `Rejected` → 0 |
| **Feature scaling** | `StandardScaler` — zero mean, unit variance |
| **Train/test split** | 80% train / 20% test, stratified by target |

> **Note:** The EDA cells and preprocessing cells each reload the CSV independently, so the notebook is safe to run in any order or re-run individual cells without corrupting the data state.

---

## Algorithms Used

Eight classification algorithms were trained and compared:

### 1. Logistic Regression
A linear model that estimates the probability of approval using a sigmoid function. Fast, interpretable, and serves as a strong baseline. Works well when the decision boundary is approximately linear.

### 2. Decision Tree
A tree-based model that splits data on feature thresholds recursively. Highly interpretable (the decision path is human-readable). Prone to overfitting without depth constraints — set to `max_depth=8` here.

### 3. Random Forest 
An ensemble of 200 decision trees, each trained on a random bootstrap sample with a random subset of features. Predictions are made by majority vote. Naturally resistant to overfitting, handles non-linear relationships, and provides reliable feature importances.

### 4. Gradient Boosting
Builds trees sequentially, where each new tree corrects the residual errors of the previous ones. Very powerful but slower to train than Random Forest. Uses 200 estimators with a default learning rate.

### 5. AdaBoost
Adaptive Boosting — assigns higher weights to misclassified samples so subsequent weak learners focus on them. Less robust to noisy data than Gradient Boosting but trains faster.

### 6. Support Vector Machine (SVM)
Finds the maximum-margin hyperplane separating the two classes in high-dimensional space. Uses an RBF (radial basis function) kernel to handle non-linear boundaries. Competitive accuracy but slower on large datasets.

### 7. K-Nearest Neighbors (KNN)
Classifies each sample based on the majority label among its 7 nearest neighbours in feature space. Simple and non-parametric but sensitive to feature scale (hence StandardScaler is essential) and slow at inference time.

### 8. Naive Bayes
Applies Bayes' theorem with the assumption that all features are conditionally independent. Very fast to train. The independence assumption is often violated in practice, which explains its lower accuracy here.

---

##  Results

All models evaluated on the held-out 20% test set with 5-fold cross-validation:

| Rank | Model | Accuracy | F1 Score | AUC-ROC | CV Mean ± Std |
|------|-------|----------|----------|---------|---------------|
| 🥇 1 | **Random Forest** | **98.36%** | **98.69%** | **99.90%** | 98.22% ± 0.38% |
| 🥈 2 | Gradient Boosting | 98.24% | 98.59% | 99.81% | 98.24% ± 0.35% |
| 🥉 3 | Decision Tree | 97.89% | 98.32% | 98.73% | 97.38% ± 0.83% |
| 4 | AdaBoost | 97.78% | 98.22% | 99.57% | 97.02% ± 0.63% |
| 5 | SVM | 94.50% | 95.56% | 98.65% | 93.89% ± 1.13% |
| 6 | Naive Bayes | 94.85% | 95.86% | 97.65% | 93.16% ± 0.81% |
| 7 | Logistic Regression | 91.22% | 93.02% | 97.26% | 91.61% ± 0.79% |
| 8 | K-Nearest Neighbors | 89.93% | 91.98% | 96.35% | 89.55% ± 1.29% |

---

## Best Model: Random Forest

Random Forest was selected as the best model based on **AUC-ROC (0.9990)** — the most reliable metric for binary classification as it measures discrimination across all thresholds.

**Why Random Forest won:**
- Highest AUC-ROC (0.9990) and accuracy (98.36%) among all models
- Most stable cross-validation score (±0.38% std — lowest variance of all models)
- No need for hyperparameter tuning to achieve near-perfect performance
- Naturally handles non-linear feature interactions present in financial data
- Provides interpretable feature importances

**Top features by importance (Random Forest):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `cibil_score` | Highest |
| 2 | `income_annum` | High |
| 3 | `loan_amount` | High |
| 4 | `bank_asset_value` | Medium |
| 5 | `luxury_assets_value` | Medium |
| 6 | `residential_assets_value` | Medium |
| 7 | `commercial_assets_value` | Low–Medium |
| 8 | `loan_term` | Low |
| 9 | `no_of_dependents` | Low |
| 10 | `education` | Low |
| 11 | `self_employed` | Lowest |

---

##  Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

##  How to Run

1. Clone or download the project folder
2. Place `loan_approval_dataset.csv` in the **same directory** as the notebook
3. Launch Jupyter:
   ```bash
   jupyter notebook loan_approval_prediction.ipynb
   ```
4. Run all cells top-to-bottom: **Kernel → Restart & Run All**

> Each section (EDA, Preprocessing, Training) reloads data independently, so individual cells can also be re-run safely without affecting others.

---

##  Evaluation Metrics Explained

| Metric | Description | Why it matters here |
|--------|-------------|---------------------|
| **Accuracy** | % of correct predictions overall | Good baseline but can be misleading on imbalanced data |
| **F1 Score** | Harmonic mean of Precision and Recall | Balances false positives (wrongly approved) and false negatives (wrongly rejected) |
| **AUC-ROC** | Area under the ROC curve (0–1) | Measures model's ability to discriminate across all thresholds; primary ranking metric |
| **CV Mean ± Std** | 5-fold cross-validation accuracy | Measures stability — low std means consistent performance on unseen data |

---


