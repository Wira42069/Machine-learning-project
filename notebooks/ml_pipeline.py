"""
Student Performance & Dropout Risk Prediction
COMP6577001 - Machine Learning Final Project
BINUS University 2025/2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier


OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)
df_raw = pd.read_csv("data/students_data.csv")
print(f"   Raw shape : {df_raw.shape}")
print(f"   Columns   : {list(df_raw.columns)}")

# Drop fully-null column and student_id (not a feature)
df = df_raw.drop(columns=["access_to_resources", "student_id"])
print(f"   After drop: {df.shape}")
print(f"   Missing   : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

## Fig 1 – Target distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Dropout risk
dropout_counts = df["dropout_risk"].value_counts()
axes[0].bar(["No Risk (0)", "At Risk (1)"], dropout_counts.values,
            color=[PALETTE[0], PALETTE[3]], edgecolor="white", linewidth=1.5)
axes[0].set_title("Dropout Risk Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(dropout_counts.values):
    axes[0].text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)

# Exam score
axes[1].hist(df["exam_score"], bins=30, color=PALETTE[1], edgecolor="white", linewidth=0.8)
axes[1].axvline(df["exam_score"].mean(), color=PALETTE[3], linestyle="--", linewidth=2,
                label=f"Mean: {df['exam_score'].mean():.1f}")
axes[1].set_title("Exam Score Distribution", fontweight="bold")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.suptitle("Target Variable Overview", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_target_distributions.png", bbox_inches="tight")
plt.close()
print("   Saved fig1_target_distributions.png")

## Fig 2 – Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("   Saved fig2_correlation_heatmap.png")

## Fig 3 – Key features vs exam score
key_features = ["study_hours_per_day", "attendance_percentage",
                "previous_gpa", "mental_health_rating", "stress_level"]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, feat in zip(axes, key_features):
    ax.scatter(df[feat], df["exam_score"], alpha=0.3, s=8, color=PALETTE[0])
    z = np.polyfit(df[feat].dropna(), df["exam_score"][df[feat].notna()], 1)
    p = np.poly1d(z)
    xs = np.linspace(df[feat].min(), df[feat].max(), 200)
    ax.plot(xs, p(xs), color=PALETTE[3], linewidth=2)
    corr_val = df[feat].corr(df["exam_score"])
    ax.set_title(f"{feat.replace('_',' ').title()}\nr={corr_val:.2f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Exam Score" if feat == key_features[0] else "")
plt.suptitle("Key Features vs Exam Score", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_features_vs_score.png", bbox_inches="tight")
plt.close()
print("   Saved fig3_features_vs_score.png")

## Fig 4 – Dropout risk by study hours + stress
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, feat in enumerate(["study_hours_per_day", "stress_level"]):
    df.boxplot(column=feat, by="dropout_risk", ax=axes[i],
               boxprops=dict(color=PALETTE[0]),
               medianprops=dict(color=PALETTE[3], linewidth=2),
               whiskerprops=dict(color=PALETTE[0]))
    axes[i].set_title(f"{feat.replace('_',' ').title()} by Dropout Risk", fontweight="bold")
    axes[i].set_xlabel("Dropout Risk (0=No, 1=Yes)")
    axes[i].set_ylabel(feat.replace("_", " ").title())
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_dropout_risk_features.png", bbox_inches="tight")
plt.close()
print("   Saved fig4_dropout_risk_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING & TRAIN/TEST SPLIT")
print("=" * 60)

FEATURES = [c for c in df.columns if c not in ["dropout_risk", "exam_score"]]
X = df[FEATURES].copy()
y_class = df["dropout_risk"].copy()
y_reg   = df["exam_score"].copy()

print(f"   Features used : {FEATURES}")
print(f"   X shape       : {X.shape}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg,   test_size=0.2, random_state=42)

print(f"   Classification train/test: {X_train_c.shape[0]} / {X_test_c.shape[0]}")
print(f"   Regression    train/test: {X_train_r.shape[0]} / {X_test_r.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION – DROPOUT RISK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CLASSIFICATION – DROPOUT RISK PREDICTION")
print("=" * 60)

class_models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                     ("clf", LogisticRegression(class_weight="balanced",
                                                                max_iter=1000, random_state=42))]),
    "Decision Tree":       Pipeline([("scaler", StandardScaler()),
                                     ("clf", DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                                    random_state=42))]),
    "Random Forest":       Pipeline([("scaler", StandardScaler()),
                                     ("clf", RandomForestClassifier(n_estimators=150, max_depth=8,
                                                                    class_weight="balanced",
                                                                    random_state=42, n_jobs=-1))]),
    "Gradient Boosting":   Pipeline([("scaler", StandardScaler()),
                                     ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                                        learning_rate=0.1,
                                                                        random_state=42))]),
    "SVM":                 Pipeline([("scaler", StandardScaler()),
                                     ("clf", SVC(kernel="rbf", class_weight="balanced",
                                                probability=True, random_state=42))]),
}

class_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in class_models.items():
    cv_scores = cross_val_score(pipe, X_train_c, y_train_c, cv=cv, scoring="roc_auc")
    pipe.fit(X_train_c, y_train_c)
    y_pred  = pipe.predict(X_test_c)
    y_proba = pipe.predict_proba(X_test_c)[:, 1]
    acc     = accuracy_score(y_test_c, y_pred)
    auc     = roc_auc_score(y_test_c, y_proba)
    class_results[name] = {
        "accuracy": acc, "roc_auc": auc,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std(),
        "y_pred": y_pred, "y_proba": y_proba, "pipe": pipe
    }
    print(f"   {name:22s} | Acc={acc:.4f} | AUC={auc:.4f} | CV_AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

best_clf_name = max(class_results, key=lambda k: class_results[k]["roc_auc"])
best_clf      = class_results[best_clf_name]
print(f"\n   ✓ Best classifier: {best_clf_name} (AUC={best_clf['roc_auc']:.4f})")

## Fig 5 – Classification comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar comparison
names  = list(class_results.keys())
aucs   = [class_results[n]["roc_auc"] for n in names]
colors = [PALETTE[3] if n == best_clf_name else PALETTE[0] for n in names]
bars   = axes[0].barh(names, aucs, color=colors, edgecolor="white")
axes[0].set_xlim(0.5, 1.0)
axes[0].axvline(0.9, color="grey", linestyle="--", linewidth=1, alpha=0.7)
axes[0].set_title("Classifier Comparison (ROC-AUC)", fontweight="bold")
axes[0].set_xlabel("ROC-AUC")
for bar, v in zip(bars, aucs):
    axes[0].text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=10)

# Confusion matrix (best model)
cm = confusion_matrix(y_test_c, best_clf["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["No Risk", "At Risk"],
            yticklabels=["No Risk", "At Risk"])
axes[1].set_title(f"Confusion Matrix\n{best_clf_name}", fontweight="bold")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# ROC curves (all models)
for i, (name, res) in enumerate(class_results.items()):
    fpr, tpr, _ = roc_curve(y_test_c, res["y_proba"])
    lw = 2.5 if name == best_clf_name else 1.2
    axes[2].plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=lw,
                 label=f"{name} ({res['roc_auc']:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[2].set_title("ROC Curves – All Classifiers", fontweight="bold")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(fontsize=8)

plt.suptitle("Classification Results – Dropout Risk", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_classification_results.png", bbox_inches="tight")
plt.close()
print("   Saved fig5_classification_results.png")

## Fig 6 – Feature importance (best clf)
best_pipe_clf = best_clf["pipe"]
if hasattr(best_pipe_clf.named_steps["clf"], "feature_importances_"):
    importances = best_pipe_clf.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_fi = [PALETTE[3] if v > feat_imp.quantile(0.75) else PALETTE[0] for v in feat_imp]
    feat_imp.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
    ax.set_title(f"Feature Importance – {best_clf_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_feature_importance_clf.png", bbox_inches="tight")
    plt.close()
    print("   Saved fig6_feature_importance_clf.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. REGRESSION – EXAM SCORE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. REGRESSION – EXAM SCORE PREDICTION")
print("=" * 60)

reg_models = {
    "Ridge Regression":    Pipeline([("scaler", StandardScaler()),
                                     ("reg", Ridge(alpha=1.0))]),
    "Lasso Regression":    Pipeline([("scaler", StandardScaler()),
                                     ("reg", Lasso(alpha=0.5, max_iter=5000))]),
    "Decision Tree":       Pipeline([("scaler", StandardScaler()),
                                     ("reg", DecisionTreeRegressor(max_depth=6, random_state=42))]),
    "Random Forest":       Pipeline([("scaler", StandardScaler()),
                                     ("reg", RandomForestRegressor(n_estimators=150, max_depth=8,
                                                                   random_state=42, n_jobs=-1))]),
    "SVR":                 Pipeline([("scaler", StandardScaler()),
                                     ("reg", SVR(kernel="rbf", C=10, epsilon=0.5))]),
}

reg_results = {}
for name, pipe in reg_models.items():
    pipe.fit(X_train_r, y_train_r)
    y_pred = pipe.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    mae  = mean_absolute_error(y_test_r, y_pred)
    r2   = r2_score(y_test_r, y_pred)
    reg_results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred, "pipe": pipe}
    print(f"   {name:22s} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

best_reg_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
best_reg      = reg_results[best_reg_name]
print(f"\n   ✓ Best regressor: {best_reg_name} (R²={best_reg['r2']:.4f})")

## Fig 7 – Regression results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar R²
names_r = list(reg_results.keys())
r2s     = [reg_results[n]["r2"] for n in names_r]
colors_r = [PALETTE[3] if n == best_reg_name else PALETTE[0] for n in names_r]
bars = axes[0].barh(names_r, r2s, color=colors_r, edgecolor="white")
axes[0].set_xlim(0, 1)
axes[0].set_title("Regressor Comparison (R²)", fontweight="bold")
axes[0].set_xlabel("R² Score")
for bar, v in zip(bars, r2s):
    axes[0].text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=10)

# Actual vs Predicted (best)
y_pred_best = best_reg["y_pred"]
axes[1].scatter(y_test_r, y_pred_best, alpha=0.4, s=10, color=PALETTE[0])
lims = [min(y_test_r.min(), y_pred_best.min()), max(y_test_r.max(), y_pred_best.max())]
axes[1].plot(lims, lims, "r--", linewidth=2)
axes[1].set_title(f"Actual vs Predicted\n{best_reg_name}", fontweight="bold")
axes[1].set_xlabel("Actual Score")
axes[1].set_ylabel("Predicted Score")

# Residuals (best)
residuals = y_test_r - y_pred_best
axes[2].scatter(y_pred_best, residuals, alpha=0.4, s=10, color=PALETTE[1])
axes[2].axhline(0, color="red", linestyle="--", linewidth=2)
axes[2].set_title(f"Residuals\n{best_reg_name}", fontweight="bold")
axes[2].set_xlabel("Predicted Score")
axes[2].set_ylabel("Residual")

plt.suptitle("Regression Results – Exam Score Prediction", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_regression_results.png", bbox_inches="tight")
plt.close()
print("   Saved fig7_regression_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SAVING MODELS")
print("=" * 60)

joblib.dump(best_clf["pipe"],  f"{MODEL_DIR}/dropout_classifier.pkl")
joblib.dump(best_reg["pipe"],  f"{MODEL_DIR}/exam_score_regressor.pkl")
joblib.dump(FEATURES,          f"{MODEL_DIR}/features.pkl")
print(f"   Saved dropout_classifier.pkl  ({best_clf_name})")
print(f"   Saved exam_score_regressor.pkl ({best_reg_name})")
print(f"   Saved features.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PRINT SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FINAL SUMMARY")
print("=" * 60)
print(f"\n  CLASSIFICATION (Dropout Risk)")
print(f"    Best model   : {best_clf_name}")
print(f"    Accuracy     : {best_clf['accuracy']:.4f}")
print(f"    ROC-AUC      : {best_clf['roc_auc']:.4f}")
print(f"    CV ROC-AUC   : {best_clf['cv_auc_mean']:.4f} ± {best_clf['cv_auc_std']:.4f}")
print(f"\n  REGRESSION (Exam Score)")
print(f"    Best model   : {best_reg_name}")
print(f"    R²           : {best_reg['r2']:.4f}")
print(f"    RMSE         : {best_reg['rmse']:.4f}")
print(f"    MAE          : {best_reg['mae']:.4f}")
print(f"\n  Classification Report ({best_clf_name}):")
print(classification_report(y_test_c, best_clf["y_pred"],
                             target_names=["No Risk", "At Risk"]))
print("=" * 60)
print("Done! All figures saved to:", OUTPUT_DIR)
