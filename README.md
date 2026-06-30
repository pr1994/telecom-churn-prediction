# 📡 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://telecom-churn-prediction-appl.streamlit.app/)

> **End-to-end machine learning project** predicting which telecom customers are likely to churn — covering EDA, preprocessing, model building, evaluation, and live deployment.

---

## 📋 Table of Contents
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key EDA Findings](#key-eda-findings)
- [Model Results](#model-results)
- [Key Business Insights](#key-business-insights)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## 💼 Business Problem

Customer churn is one of the most costly challenges in the telecom industry. Acquiring a new customer costs **5–10x more** than retaining an existing one. This project builds a machine learning model to **proactively identify customers at high risk of churning**, enabling the business to intervene before they leave.

**Core Question:**
> *"Given a customer's usage patterns, plan details, and service history — will they churn?"*

---

## 📦 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Orange Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) |
| Author | mnassrib |
| Train Set | 2,666 rows × 20 columns |
| Test Set | 667 rows × 20 columns |
| Target Variable | `Churn` (True / False) |
| Churn Rate | ~14% (Class Imbalanced) |

**Key Features:**

| Feature | Description |
|---|---|
| `Customer service calls` | Number of times customer contacted support |
| `International plan` | Whether customer has an international plan |
| `Voice mail plan` | Whether customer has a voicemail plan |
| `Total day minutes` | Total daytime call minutes |
| `Total eve/night minutes` | Evening and night call usage |
| `Account length` | Duration of customer relationship |

---

## 📁 Project Structure

```
telecom-churn-prediction/
│
├── telecom_churn_analysis.ipynb   # Main Colab notebook (full pipeline)
├── app.py                         # Streamlit web application
├── rf_model.pkl                   # Trained Random Forest model
├── scaler.pkl                     # Fitted StandardScaler
├── requirements.txt               # Deployment dependencies
├── data/churn-bigml-80.csv             # Training set
├── data/churn-bigml-20.csv             # Test set
└── README.md                      # Project documentation
```

---

## 🔬 Methodology

The project follows a structured 6-phase ML pipeline:

```
Phase 1 → Exploratory Data Analysis (EDA)
Phase 2 → Data Preprocessing
Phase 3 → Model Building
Phase 4 → Model Evaluation
Phase 5 → Feature Importance & Business Insights
Phase 6 → Deployment (Streamlit Cloud)
```

---

## 📊 Key EDA Findings

### 1. Class Imbalance
~14% of customers churned vs 86% who stayed. This makes **accuracy a misleading metric** — a model predicting "No Churn" always would still achieve 86% accuracy. Recall and F1-Score were used as primary evaluation metrics.

### 2. Multicollinearity Detected
Four pairs of highly correlated features were identified and resolved:

| Redundant Pair | Action |
|---|---|
| Total Day Minutes ↔ Total Day Charge | Dropped charge column |
| Total Eve Minutes ↔ Total Eve Charge | Dropped charge column |
| Total Night Minutes ↔ Total Night Charge | Dropped charge column |
| Total Intl Minutes ↔ Total Intl Charge | Dropped charge column |

> Charge columns are mathematically derived from minutes (Charge = Minutes × Rate), making them redundant. Keeping both would introduce multicollinearity and mislead the model.

### 3. Strong Churn Signals Identified

| Signal | Finding |
|---|---|
| Customer Service Calls ≥ 5 | Churn rate spikes dramatically |
| International Plan = Yes | Significantly higher churn rate |
| High Total Day Minutes | Churned customers use more daytime minutes |
| Voice Mail Plan = Yes | Associated with lower churn |

---

## 🤖 Model Results

Four models were built with increasing sophistication:

| Model | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.179 | 0.254 | 0.825 |
| Logistic Regression (Balanced) | 0.758 | 0.488 | 0.829 |
| XGBoost | 0.674 | 0.685 | 0.852 |
| **Random Forest (Balanced) ✅** | **0.590** | **0.737** | **0.920** |

### 🏆 Best Model: Random Forest (class_weight=balanced)

Random Forest was selected as the final model based on:
- **Highest ROC-AUC (0.920)** — best overall ability to separate churners from non-churners
- **Highest F1-Score (0.737)** — best balance between catching churners and avoiding false alarms
- More reliable and actionable predictions for business use

> **Why not LR Balanced despite higher recall?**
> LR Balanced's recall of 0.758 comes at the cost of too many false alarms (F1 of 0.488). In practice, flagging too many loyal customers as churners wastes retention resources and erodes trust in the model.

### Top Features by Importance (Random Forest)

```
1. Total Day Minutes         ████████████████████  (Highest)
2. Customer Service Calls    ████████████████
3. International Plan        ████████████
4. Total Intl Minutes        ████████
5. Account Length            ██████
```

> These match exactly with what EDA identified manually — confirming the model learned genuine patterns, not noise.

---

## 💡 Key Business Insights

| Insight | Recommended Action |
|---|---|
| Customers with 4+ service calls are high-risk | Flag for immediate proactive outreach |
| International plan holders churn at higher rates | Review international pricing competitiveness |
| High daytime usage customers are churning | Priority targets for retention offers |
| Voicemail plan reduces churn | Promote voicemail adoption as retention strategy |

---

## ▶️ How to Run

### Option 1 — Live Demo (No setup needed)
👉 [Launch the app directly](https://telecom-churn-prediction-appl.streamlit.app/)

### Option 2 — Run Locally
```bash
git clone https://github.com/pr1994/telecom-churn-prediction.git
cd telecom-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

### Option 3 — Explore the Notebook
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `telecom_churn_analysis.ipynb`
3. Upload both CSV files to the Colab session (`/content/`)
4. Run all cells top to bottom (`Runtime → Run All`)

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Environment:** Google Colab + Local
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Deployment:** Streamlit Cloud

---

## 👤 Author

**Pritam**
IT Professional (10 years Oracle WebCenter Content) → Transitioning to AI/ML

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/pr1994)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/pritambiswas-wcc/)

---
