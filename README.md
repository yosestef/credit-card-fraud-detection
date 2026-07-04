# Credit Card Fraud Detection

Fraud detection system using machine learning on imbalanced transaction data, with SQL data handling and Power BI dashboard visualization.

## Table of contents

- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Handling class imbalance](#handling-class-imbalance)
- [Model](#model)
- [Dashboard](#dashboard)
- [Project structure](#project-structure)
- [How to run this project](#how-to-run-this-project)
- [Key findings](#key-findings)
- [Next steps](#next-steps)

## Project overview

This project builds a fraud detection pipeline on real, anonymized credit card transactions. The main challenge tackled here is **extreme class imbalance**: only 0.17% of transactions in the dataset are fraudulent, which means a naive model could reach 99.8% accuracy while never catching a single fraud case.

The project covers the full workflow: exploratory data analysis, class imbalance handling, model training and evaluation with the right metrics, and a Power BI dashboard for presenting results to a non-technical audience.

## Dataset

- **Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns
- **Features:** `Time`, `Amount`, and 28 anonymized PCA-transformed features (`V1`–`V28`)
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- **Missing values:** none

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |

## Tech stack

| Tool | Purpose |
|---|---|
| Python (pandas, scikit-learn) | Data manipulation and modeling |
| matplotlib, seaborn | Exploratory visualization |
| imbalanced-learn (SMOTE) | Class imbalance handling |
| SQL / MySQL | Data storage and querying |
| Power BI | Executive dashboard |
| Jupyter Notebook | Development environment |

## Exploratory data analysis

### Class distribution

The dataset is heavily imbalanced, which is the central challenge addressed throughout this project.

![Class distribution](images/01_class_distribution.png)

### Transaction amount distribution

Transaction amounts are right-skewed: most transactions are low-value, with a long tail of high-value outliers.

![Amount distribution](images/02_amount_distribution.png)

### Amount by class

Fraudulent transactions tend to have higher and more variable amounts than legitimate ones.

![Amount by class](images/03_amount_by_class.png)

### Transactions by hour

![Transactions by hour](images/04_transactions_by_hour.png)

*Fraudulent transactions show two anomalous peaks around 2:00 AM and 11:00 AM that do not match the distribution of legitimate activity, suggesting automated or coordinated attack patterns during low-monitoring hours*

## Handling class imbalance

Two complementary strategies were used:

1. **`class_weight='balanced'`** in the model, so misclassifying the minority class is penalized more heavily.
2. **SMOTE (Synthetic Minority Oversampling Technique)**, applied **only to the training set** (never to the test set), to generate synthetic fraud examples and give the model more minority-class patterns to learn from.

```
raw data
   |
   v
train/test split   <- split first
   |
   v
SMOTE on train only   <- balance only the training data
   |
   v
train model
   |
   v
evaluate on original (untouched) test set
```

Accuracy is not used as an evaluation metric here, since it is misleading on imbalanced data. Instead, the model is evaluated with:

- **Precision** — of all transactions flagged as fraud, how many actually were
- **Recall** — of all real fraud cases, how many were caught
- **F1-score** — balance between precision and recall
- **AUC-ROC** — overall ability to separate the two classes

## Model

*(To be completed in Day 3 — model choice, training process, and evaluation results.)*

## Dashboard

*(To be completed in Day 4 — Power BI dashboard screenshots and description.)*

## Project structure

```
fraud-detection-project/
├── data/                  # raw dataset (not tracked in git, see .gitignore)
├── notebooks/
│   └── 01_eda.ipynb       # exploratory data analysis
├── src/                   # reusable scripts (future)
├── images/                # charts and screenshots used in this README
├── requirements.txt
└── README.md
```

## How to run this project

```bash
# clone the repo
git clone https://github.com/YOUR-USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# install dependencies
pip install -r requirements.txt

# download the dataset from Kaggle and place it in data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# launch Jupyter
jupyter notebook
```

## Key findings

- The dataset is clean, with no missing values across any of the 31 columns.
- Fraudulent transactions represent only 0.17% of the data, requiring specific techniques (SMOTE, class weighting) rather than a standard classification approach.
- Fraudulent transactions tend to involve higher and more variable amounts than legitimate ones.

## Next steps

- [ ] Train and evaluate a Random Forest classifier
- [ ] Compare performance with and without SMOTE
- [ ] Build the Power BI dashboard
- [ ] Document final results and conclusions
