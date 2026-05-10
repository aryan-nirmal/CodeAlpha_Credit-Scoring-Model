# 🏦 Credit Scoring Model

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6DB33F?style=for-the-badge)](https://imbalanced-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![CodeAlpha](https://img.shields.io/badge/Internship-CodeAlpha-blueviolet?style=for-the-badge)](https://codealpha.tech)

> **Task 1 — CodeAlpha Machine Learning Internship**  
> A production-ready credit risk classification pipeline that predicts whether a borrower will experience serious financial distress within two years, using the *Give Me Some Credit* dataset from Kaggle.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Results](#-results)
- [Plots](#-plots)
- [Quick Start](#-quick-start)
- [Predict on New Data](#-predict-on-new-data)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

Credit scoring is a fundamental problem in consumer finance. This project builds and compares three machine learning classifiers to predict the probability that a borrower will experience **90+ days of financial distress** within the next 2 years — allowing lenders to make better-informed credit decisions.

**Key highlights:**
- Handles severe class imbalance (~93 % / 7 %) using **SMOTE** oversampling
- Engineered 5 domain-informed features on top of the 10 original columns
- Benchmarks **Logistic Regression**, **Random Forest**, and **XGBoost**
- Saves the best model (by ROC-AUC) for immediate inference
- Includes an interactive CLI predictor with risk-tier assessment

---

## 📂 Dataset

| Property | Value |
|---|---|
| **Name** | Give Me Some Credit |
| **Source** | [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit) |
| **File** | `cs-training.csv` |
| **Rows** | 150,000 |
| **Features** | 10 original + 5 engineered |
| **Target** | `SeriousDlqin2yrs` (0 = no default, 1 = default) |
| **Class balance** | ~93 % / 7 % (severe imbalance) |

Download `cs-training.csv` from Kaggle and place it in the project root before running.

### Original Feature Descriptions

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards / credit limits |
| `age` | Age of the borrower in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days past due in last 2 years |
| `DebtRatio` | Monthly debt payments / monthly gross income |
| `MonthlyIncome` | Monthly gross income (USD) |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines + installment loans |
| `NumberOfTimes90DaysLate` | Times 90+ days overdue in last 2 years |
| `NumberRealEstateLoansOrLines` | Mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days past due in last 2 years |
| `NumberOfDependents` | Number of dependents (spouse, children, etc.) |

---

## 🗂 Project Structure

```
CodeAlpha_CreditScoringModel/
├── credit_scoring.py          # Main pipeline: EDA → preprocess → train → evaluate → save
├── predict_custom.py          # Interactive CLI for predicting new applicants
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                 # Files/dirs excluded from version control
│
├── plots/                     # Auto-generated visualisations
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── metrics_comparison.png
│
└── saved_model/               # Persisted model bundle
    └── best_credit_model.pkl  # Best model + scaler + feature names
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                             │
│                                                                  │
│  cs-training.csv                                                 │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Load Data  │───▶│Feature Engineer │───▶│  Preprocessing  │  │
│  │  (150k rows)│    │  +5 new feats   │    │ • Impute median │  │
│  └─────────────┘    └─────────────────┘    │ • Cap outliers  │  │
│                                            │ • Train/Test 80/│  │
│                                            │ • StandardScaler│  │
│                                            │ • SMOTE balance │  │
│                                            └────────┬────────┘  │
│                                                     │           │
│                              ┌──────────────────────┤           │
│                              ▼                      ▼           │
│                   ┌─────────────────┐    ┌─────────────────┐   │
│                   │ X_train (SMOTE) │    │     X_test      │   │
│                   │ balanced classes│    │  (stratified)   │   │
│                   └────────┬────────┘    └────────┬────────┘   │
│                            │                      │             │
│          ┌─────────────────┼─────────────────┐    │             │
│          ▼                 ▼                 ▼    │             │
│   ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │             │
│   │  Logistic   │  │   Random    │  │ XGBoost  │ │             │
│   │ Regression  │  │   Forest    │  │          │ │             │
│   └──────┬──────┘  └──────┬──────┘  └────┬─────┘ │             │
│          └────────────────┼──────────────┘       │             │
│                           ▼ predict              ▼             │
│                   ┌───────────────────────────────────┐        │
│                   │         Evaluation                │        │
│                   │  Accuracy | Precision | Recall    │        │
│                   │  F1-Score | ROC-AUC               │        │
│                   └───────────────┬───────────────────┘        │
│                                   │                             │
│                   ┌───────────────┼──────────────────┐         │
│                   ▼               ▼                  ▼         │
│           ┌──────────────┐ ┌────────────┐  ┌───────────────┐   │
│           │   Plots/     │ │ Best Model │  │ predict_      │   │
│           │   5 charts   │ │  .pkl save │  │ custom.py CLI │   │
│           └──────────────┘ └────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Feature Engineering

Five new features are derived from the raw columns to improve model signal:

| Feature | Formula | Intuition |
|---|---|---|
| `DebtToIncome` | `DebtRatio × MonthlyIncome` | Absolute monthly debt burden |
| `IncomePerDependent` | `MonthlyIncome / (Dependents + 1)` | Effective disposable income per person |
| `TotalPastDue` | Sum of all past-due columns | Overall delinquency history |
| `UtilRatio_Age` | `RevolvingUtil × log1p(age)` | Age-weighted credit stress indicator |
| `HighUtilization` | `RevolvingUtil > 0.75` (binary) | Flag for dangerously high credit usage |

---

## 🤖 Models

### Logistic Regression
- Baseline linear model with L2 regularisation (`C=0.5`)
- `class_weight="balanced"` to handle imbalance
- Provides interpretable coefficients

### Random Forest
- 200 decision trees, `max_depth=12`, `min_samples_leaf=10`
- Ensemble averaging reduces variance
- Feature importances via Gini impurity

### XGBoost
- 300 gradient-boosted trees, `learning_rate=0.05`
- `scale_pos_weight=13` to handle class imbalance natively
- Column subsampling & row subsampling for regularisation

---

## 📊 Results

> Evaluated on a held-out stratified 20 % test set (30,000 samples).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7521 | 0.2634 | 0.7289 | 0.3863 | 0.8412 |
| Random Forest | 0.8934 | 0.5812 | 0.5631 | 0.5720 | 0.8743 |
| **XGBoost** ⭐ | **0.9087** | **0.6241** | **0.5912** | **0.6072** | **0.8891** |

> ⭐ XGBoost achieves the highest ROC-AUC and is saved as the best model.

**Key observations:**
- Logistic Regression has high recall (catches more defaults) but low precision (many false alarms)
- Random Forest and XGBoost achieve a better precision-recall balance
- XGBoost edges out Random Forest on all tree-based metrics, especially ROC-AUC
- SMOTE significantly improved recall for the minority (default) class across all models

---

## 📈 Plots

All plots are saved to the `plots/` directory after running `credit_scoring.py`.

| Plot | Description |
|---|---|
| `correlation_heatmap.png` | Feature correlation matrix (lower triangle) |
| `roc_curves.png` | All three ROC curves overlaid on one chart |
| `confusion_matrices.png` | Confusion matrices side-by-side for all models |
| `feature_importance.png` | Top features — coefficients for LR, Gini for RF & XGB |
| `metrics_comparison.png` | Grouped bar chart of all 5 metrics across models |

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `cs-training.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in the project root:

```
CodeAlpha_CreditScoringModel/
└── cs-training.csv   ← here
```

### 5. Run the pipeline

```bash
python credit_scoring.py
```

This will:
- Load and preprocess the data
- Engineer new features
- Train all three models with SMOTE
- Print evaluation metrics
- Save 5 plots to `plots/`
- Save the best model to `saved_model/best_credit_model.pkl`

---

## 🔮 Predict on New Data

### Interactive mode (manual input)

```bash
python predict_custom.py
```

You'll be prompted to enter each feature value. Press **Enter** to accept the default.

### Demo mode (preset values)

```bash
python predict_custom.py --demo
```

### Example output

```
══════════════════════════════════════════════════════════════
  PREDICTION RESULT
══════════════════════════════════════════════════════════════
  Model used            : XGBoost
  Probability of default: 8.34 %
  Verdict               : CREDITWORTHY ✔
──────────────────────────────────────────────────────────────

  Risk gauge (green = safe, red = risky)
  [█████████████████████████░░░░░]
  Safe: 91.7 %  |  Risky: 8.3 %

  Assessment : LOW RISK — Strong candidate for credit.
══════════════════════════════════════════════════════════════
```

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical operations |
| scikit-learn | 1.3+ | ML algorithms & metrics |
| imbalanced-learn | 0.11+ | SMOTE oversampling |
| XGBoost | 2.0+ | Gradient boosting |
| Matplotlib | 3.7+ | Plotting |
| Seaborn | 0.12+ | Statistical visualisation |
| joblib | 1.3+ | Model serialisation |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Kaggle / FICO** for the *Give Me Some Credit* dataset
- **CodeAlpha** for the internship opportunity
- The open-source communities behind scikit-learn, XGBoost, and imbalanced-learn

---

<p align="center">Made with ❤️ during the CodeAlpha ML Internship</p>
