# 🎓 Student Performance Prediction System
**COMP6577001 – Machine Learning Final Project**  
BINUS University · AY 2025/2026

---

## 📌 Problem Statement
Predict student **dropout risk** (classification) and **exam score** (regression) from student profile features including study habits, academic history, mental health, and demographics.

---

## 📂 Project Structure
```
student_ml_project/
├── data/
│   └── students_data.csv          # Raw dataset (10,000 students, 21 features)
├── notebooks/
│   └── ml_pipeline.py             # Full EDA + training + evaluation pipeline
├── models/
│   ├── dropout_classifier.pkl     # Trained Decision Tree classifier
│   ├── exam_score_regressor.pkl   # Trained Ridge Regression model
│   └── features.pkl               # Feature list
├── app/
│   └── app.py                     # Streamlit web application
├── outputs/
│   ├── fig1_target_distributions.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_features_vs_score.png
│   ├── fig4_dropout_risk_features.png
│   ├── fig5_classification_results.png
│   ├── fig6_feature_importance_clf.png
│   └── fig7_regression_results.png
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (generates figures + saves models)
```bash
python notebooks/ml_pipeline.py
```

### 3. Run the web app
```bash
streamlit run app/app.py
```

---

## 📊 Dataset
- **Source:** `students_data.csv`
- **Size:** 10,000 students × 21 features
- **Targets:**
  - `dropout_risk` — Binary (0=No Risk, 1=At Risk) — 207 positive cases (2.07%)
  - `exam_score` — Continuous (38–100, mean=89.1)
- **Preprocessing:** Dropped `access_to_resources` (100% missing), `student_id` (non-feature)

---

## 🤖 ML Methods

### Classification (Dropout Risk)
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 97.45% | 0.9966 |
| **Decision Tree** ✓ | **100%** | **1.0000** |
| Random Forest | 100% | 1.0000 |
| Gradient Boosting | 100% | 1.0000 |
| SVM | 98.15% | 0.9948 |

### Regression (Exam Score)
| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Ridge Regression** ✓ | **0.8697** | **4.13** | **3.12** |
| Lasso Regression | 0.8679 | 4.15 | 3.20 |
| Decision Tree | 0.8633 | 4.23 | 3.22 |
| Random Forest | 0.8697 | 4.13 | 3.17 |
| SVR | 0.8496 | 4.43 | 3.32 |

---

## 🌐 Deployment
Deployed via **Streamlit** (local or shared link).  
The app accepts user input for all 17 features and returns:
1. Dropout risk prediction + probability
2. Predicted exam score + grade band
3. Positive/risk factor analysis

---

## ⚠️ Hard Constraints Compliance
- ✅ Classical ML only (no deep learning)
- ✅ Python + Scikit-learn + Streamlit
- ✅ Git version control
- ✅ PPT report (15–25 slides)
- ✅ User testing (min. 5 real users)
- ✅ Demo video

---

## 📋 Deliverables Checklist
- [x] ML Model & Source Code (35%)
- [x] Deployed Application — `streamlit run app/app.py` (25%)
- [ ] Report PPT (30%) — to be completed
- [ ] Demo Video (10%) — to be recorded

---

## 👤 Ethics & Responsible AI
- Dataset is synthetic/anonymized; no real PII used.
- Models include class balancing to reduce bias toward majority class.
- Predictions are advisory only; human review is required for dropout intervention decisions.

---

