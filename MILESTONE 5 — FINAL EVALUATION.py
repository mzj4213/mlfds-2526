# =========================================================
# MILESTONE 5 — FINAL EVALUATION
# KD34403 Machine Learning for Data Science
# Final Testing Metrics & Findings
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# =========================================================
# PRELIMINARY SETUP
# =========================================================

SEED = 42
np.random.seed(SEED)

# =========================================================
# LOAD DATA FROM MILESTONE 1
# =========================================================

print("Loading processed dataset files...")

X_train = np.load(r'D:\Python Program\X_train.npy')
X_test = np.load(r'D:\Python Program\X_test.npy')

y_train = np.load(r'D:\Python Program\y_train.npy')
y_test = np.load(r'D:\Python Program\y_test.npy')

# =========================================================
# LOAD ENCODED FEATURE NAMES
# =========================================================

with open(r'D:\Python Program\feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

print("Data loaded successfully!")

print("\nDataset Summary")
print("----------------------------")
print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])
print("Number of Features:", X_train.shape[1])

# =========================================================
# FINAL TUNED MODEL
# (Using Optimized Parameters from Milestone 4)
# =========================================================

print("\nBuilding final tuned XGBoost model...")

# Recalculate class imbalance weight
n_class_0 = (y_train == 0).sum()
n_class_1 = (y_train == 1).sum()

scale_pos_weight = n_class_0 / n_class_1

print(f"\nClass Imbalance Ratio: {scale_pos_weight:.2f}:1")

# Final tuned model
model = XGBClassifier(

    # Optimized parameters from Milestone 4
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.5,

    # Imbalance handling
    scale_pos_weight=scale_pos_weight,

    # General setup
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=SEED
)

# =========================================================
# TRAIN FINAL MODEL
# =========================================================

print("\nTraining final optimized model...")

model.fit(
    X_train,
    y_train,
    verbose=False
)

print("Final model trained successfully!")

# =========================================================
# TEST SET PREDICTIONS
# =========================================================

print("\nEvaluating model on unseen test data...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================================
# FINAL TEST METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n=================================================")
print("FINAL TEST RESULTS")
print("=================================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n=================================================")
print("CLASSIFICATION REPORT")
print("=================================================")

print(classification_report(
    y_test,
    y_pred,
    target_names=['No Subscription', 'Subscription']
))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n=================================================")
print("CONFUSION MATRIX DETAILS")
print("=================================================")

print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

# =========================================================
# VISUALIZATION 1 — CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Subscription', 'Subscription'],
    yticklabels=['No Subscription', 'Subscription']
)

plt.title('Final Model Confusion Matrix', fontsize=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 2 — ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'ROC Curve (AUC = {roc_auc:.4f})'
)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.title('ROC Curve', fontsize=15)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 3 — PRECISION-RECALL CURVE
# =========================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2
)

plt.title('Precision-Recall Curve', fontsize=15)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)

plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 4 — FINAL METRICS BAR CHART
# =========================================================

metrics = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1-Score',
    'ROC-AUC'
]

scores = [
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    metrics,
    scores,
    color=[
        'skyblue',
        'lightgreen',
        'orange',
        'violet',
        'red'
    ]
)

plt.ylim(0, 1)

plt.title('Final Evaluation Metrics', fontsize=15)
plt.ylabel('Score', fontsize=12)

# Add labels above bars
for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.01,
        f'{yval:.4f}',
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 5 — CLASS DISTRIBUTION
# =========================================================

class_counts = pd.Series(y_test).value_counts()

plt.figure(figsize=(6, 5))

plt.bar(
    ['No Subscription', 'Subscription'],
    class_counts.values,
    color=['steelblue', 'orange']
)

plt.title('Test Set Class Distribution', fontsize=15)
plt.ylabel('Number of Samples')

for i, v in enumerate(class_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 6 — FEATURE IMPORTANCE
# =========================================================

feature_importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort importance values
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

# Select Top 10 Features
top_features = importance_df.head(10)

# Plot
plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=top_features
)

plt.title('Top 10 Feature Importances', fontsize=15)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)

plt.tight_layout()
plt.show()

print("Milestone 5 completed successfully!")