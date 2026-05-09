# Milestone 4 : Model Optimization
import numpy as np
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, log_loss, cohen_kappa_score
)
import random
import pandas as pd

# Preliminary Setup
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 1. Load the data from Milestone 1
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

print("Data loaded successfully!")

# 2. Model Setup

xgb_base = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=SEED, 
    eval_metric='logloss',
    objective="binary:logistic",
)

# class weights implementation
n_samples = len(y_train)
n_class_0 = (y_train == 0).sum()      # "No Subscription"
n_class_1 = (y_train == 1).sum()      # "Subscription"

scale_pos_weight = n_class_0 / n_class_1  # Weight for minority class

print(f"\nDataset Composition:")
print(f"├─ Total samples: {n_samples}")
print(f"├─ Class 0 (No Subscription): {n_class_0} ({n_class_0/n_samples*100:.1f}%)")
print(f"├─ Class 1 (Subscription): {n_class_1} ({n_class_1/n_samples*100:.1f}%)")
print(f"└─ Imbalance Ratio: {n_class_0/n_class_1:.2f}:1")

print(f"\nClass Weight Calculation:")
print(f"├─ Formula: scale_pos_weight = n_class_0 / n_class_1")
print(f"├─ Calculation: {n_class_0} / {n_class_1}")
print(f"└─ Result: scale_pos_weight = {scale_pos_weight:.2f}")
print(f"   (Penalize minority class errors {scale_pos_weight:.2f}x more)")



xgb_tuned = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=SEED,
    eval_metric='logloss',
    early_stopping_rounds=10,  # Stop if val metric doesn't improve
    scale_pos_weight=scale_pos_weight,  
)

# 3. Model Training
xgb_base.fit(X_train, y_train)
xgb_base_acc = accuracy_score(y_val, xgb_base.predict(X_val))



# Train with eval_set to monitor validation performance
xgb_tuned.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
xgb_tuned_acc = accuracy_score(y_val, xgb_tuned.predict(X_val))

# 4. Output Results
print(f"XGBoost Accuracy: {xgb_base_acc:.4f}")

# Compare metrics
xgb_base_train_pred = xgb_base.predict(X_train)
xgb_base_val_pred = xgb_base.predict(X_val)
print("Training Accuracy:", accuracy_score(y_train, xgb_base_train_pred))
print("Validation Accuracy:", accuracy_score(y_val, xgb_base_val_pred))
print("\nOVERFITTING INDICATOR: If train >> validation, overfitting is severe")


print(f"Tuned XGBoost Accuracy: {xgb_tuned_acc:.4f}")
print(f"Best iteration: {xgb_tuned.best_iteration}")  # e.g., 47
print(f"Best logloss: {xgb_tuned.best_score:.4f}")    # e.g., 0.3892

# Compare metrics
xgb_tuned_train_pred = xgb_tuned.predict(X_train)
xgb_tuned_val_pred = xgb_tuned.predict(X_val)
print("Training Accuracy:", accuracy_score(y_train, xgb_tuned_train_pred))
print("Validation Accuracy:", accuracy_score(y_val, xgb_tuned_val_pred))
print("\nOVERFITTING INDICATOR: If train >> validation, overfitting is severe")

# 5. COMPREHENSIVE VALIDATION METRICS

print("\n" + "="*70)
print("DETAILED MODEL EVALUATION & COMPARISON")
print("="*70)

# ==================== MODEL 1: BASELINE ====================
print("\n📊 MODEL 1: BASELINE XGBoost")
print("-" * 70)

xgb_base_pred_proba = xgb_base.predict_proba(X_val)[:, 1]  # Get probability for class 1

# Classification Metrics
xgb_base_accuracy = accuracy_score(y_val, xgb_base_val_pred)
xgb_base_precision = precision_score(y_val, xgb_base_val_pred)
xgb_base_recall = recall_score(y_val, xgb_base_val_pred)
xgb_base_f1 = f1_score(y_val, xgb_base_val_pred)
xgb_base_roc_auc = roc_auc_score(y_val, xgb_base_pred_proba)
xgb_base_logloss = log_loss(y_val, xgb_base_pred_proba)

print(f"Accuracy:         {xgb_base_accuracy:.4f}")
print(f"Precision:        {xgb_base_precision:.4f}")
print(f"Recall:           {xgb_base_recall:.4f}")
print(f"F1-Score:         {xgb_base_f1:.4f}")
print(f"ROC-AUC:          {xgb_base_roc_auc:.4f}")
print(f"Log Loss:         {xgb_base_logloss:.4f}")

# Overfitting Check
xgb_base_train_proba = xgb_base.predict_proba(X_train)[:, 1]
xgb_base_train_logloss = log_loss(y_train, xgb_base_train_proba)
xgb_base_overfit_gap = xgb_base_train_logloss - xgb_base_logloss

print(f"\nTraining Log Loss: {xgb_base_train_logloss:.4f}")
print(f"Validation Log Loss: {xgb_base_logloss:.4f}")
print(f"Overfitting Gap:  {xgb_base_overfit_gap:.4f}")

# Confusion Matrix
xgb_base_cm = confusion_matrix(y_val, xgb_base_val_pred)
xgb_base_tn, xgb_base_fp, xgb_base_fn, xgb_base_tp = xgb_base_cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {xgb_base_tn}")
print(f"  False Positives: {xgb_base_fp}")
print(f"  False Negatives: {xgb_base_fn}")
print(f"  True Positives:  {xgb_base_tp}")

# Specificity & Sensitivity
xgb_base_specificity = xgb_base_tn / (xgb_base_tn + xgb_base_fp)
xgb_base_sensitivity = xgb_base_tp / (xgb_base_tp + xgb_base_fn)
print(f"\nSensitivity (True Positive Rate): {xgb_base_sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {xgb_base_specificity:.4f}")

print("\nClassification Report (Baseline):")
print(classification_report(y_val, xgb_base_val_pred, 
      target_names=['No Subscription', 'Subscription']))


# ==================== MODEL 2: TUNED ====================
print("\n" + "="*70)
print("📊 MODEL 2: TUNED XGBoost (with Early Stopping & Regularization)")
print("-" * 70)

xgb_tuned_pred_proba = xgb_tuned.predict_proba(X_val)[:, 1]

# Classification Metrics
xgb_tuned_accuracy = accuracy_score(y_val, xgb_tuned_val_pred)
xgb_tuned_precision = precision_score(y_val, xgb_tuned_val_pred)
xgb_tuned_recall = recall_score(y_val, xgb_tuned_val_pred)
xgb_tuned_f1 = f1_score(y_val, xgb_tuned_val_pred)
xgb_tuned_roc_auc = roc_auc_score(y_val, xgb_tuned_pred_proba)
xgb_tuned_logloss = log_loss(y_val, xgb_tuned_pred_proba)

print(f"Accuracy:         {xgb_tuned_accuracy:.4f}")
print(f"Precision:        {xgb_tuned_precision:.4f}")
print(f"Recall:           {xgb_tuned_recall:.4f}")
print(f"F1-Score:         {xgb_tuned_f1:.4f}")
print(f"ROC-AUC:          {xgb_tuned_roc_auc:.4f}")
print(f"Log Loss:         {xgb_tuned_logloss:.4f}")

# Overfitting Check
xgb_tuned_train_proba = xgb_tuned.predict_proba(X_train)[:, 1]
xgb_tuned_train_logloss = log_loss(y_train, xgb_tuned_train_proba)
xgb_tuned_overfit_gap = xgb_tuned_train_logloss - xgb_tuned_logloss

print(f"\nTraining Log Loss: {xgb_tuned_train_logloss:.4f}")
print(f"Validation Log Loss: {xgb_tuned_logloss:.4f}")
print(f"Overfitting Gap:  {xgb_tuned_overfit_gap:.4f}")
print(f"Best Iteration:   {xgb_tuned.best_iteration}")
print(f"Early Stopping Saved {500 - xgb_tuned.best_iteration} iterations")

# Confusion Matrix
xgb_tuned_cm = confusion_matrix(y_val, xgb_tuned_val_pred)
xgb_tuned_tn, xgb_tuned_fp, xgb_tuned_fn, xgb_tuned_tp = xgb_tuned_cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {xgb_tuned_tn}")
print(f"  False Positives: {xgb_tuned_fp}")
print(f"  False Negatives: {xgb_tuned_fn}")
print(f"  True Positives:  {xgb_tuned_tp}")

# Specificity & Sensitivity
xgb_tuned_specificity = xgb_tuned_tn / (xgb_tuned_tn + xgb_tuned_fp)
xgb_tuned_sensitivity = xgb_tuned_tp / (xgb_tuned_tp + xgb_tuned_fn)
print(f"\nSensitivity (True Positive Rate): {xgb_tuned_sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {xgb_tuned_specificity:.4f}")

print("\nClassification Report (Tuned):")
print(classification_report(y_val, xgb_tuned_val_pred, 
      target_names=['No Subscription', 'Subscription']))


# ==================== MODEL COMPARISON ====================
print("\n" + "="*70)
print("🔄 MODEL COMPARISON")
print("="*70)

comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 
               'Log Loss', 'Overfitting Gap'],
    'Baseline': [xgb_base_accuracy, xgb_base_precision, xgb_base_recall, xgb_base_f1, 
                 xgb_base_roc_auc, xgb_base_logloss, xgb_base_overfit_gap],
    'Tuned': [xgb_tuned_accuracy, xgb_tuned_precision, xgb_tuned_recall, 
              xgb_tuned_f1, xgb_tuned_roc_auc, xgb_tuned_logloss, 
              xgb_tuned_overfit_gap]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Difference'] = comparison_df['Tuned'] - comparison_df['Baseline']
comparison_df['Winner'] = comparison_df['Difference'].apply(
    lambda x: '🎯 Tuned' if abs(x) > 0.001 and (x > 0 if comparison_df.index[comparison_df['Difference'] == x].tolist()[0] != 7 else x < 0) else 'Baseline' if x < -0.001 else '='
)

print("\n" + comparison_df.to_string(index=False))


# ==================== VISUALIZATIONS ====================

# 1. Confusion Matrix Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(xgb_base_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['No Subscription', 'Subscription'], yticklabels=['No Subscription', 'Subscription'])
axes[0].set_title('Baseline Model - Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

sns.heatmap(xgb_tuned_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Subscription', 'Subscription'], yticklabels=['No Subscription', 'Subscription'])
axes[1].set_title('Tuned Model - Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()


# 2. ROC Curve Comparison
plt.figure(figsize=(10, 8))

xgb_base_fpr, xgb_base_tpr, _ = roc_curve(y_val, xgb_base_pred_proba)
xgb_tuned_fpr, xgb_tuned_tpr, _ = roc_curve(y_val, xgb_tuned_pred_proba)

plt.plot(xgb_base_fpr, xgb_base_tpr, label=f'Baseline (AUC = {xgb_base_roc_auc:.4f})', linewidth=2)
plt.plot(xgb_tuned_fpr, xgb_tuned_tpr, label=f'Tuned (AUC = {xgb_tuned_roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# 4. Metrics Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ('Accuracy', [xgb_base_accuracy, xgb_tuned_accuracy]),
    ('Precision', [xgb_base_precision, xgb_tuned_precision]),
    ('Recall', [xgb_base_recall, xgb_tuned_recall]),
    ('F1-Score', [xgb_base_f1, xgb_tuned_f1])
]

for idx, (metric_name, values) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(['Baseline', 'Tuned'], values, color=['skyblue', 'lightgreen'])
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'{metric_name} Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# 5. Overfitting Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = ['Baseline', 'Tuned']
train_loss = [xgb_base_train_logloss, xgb_tuned_train_logloss]
val_loss = [xgb_base_logloss, xgb_tuned_logloss]

x = np.arange(len(models))
width = 0.35

bars1 = axes[0].bar(x - width/2, train_loss, width, label='Training Loss', color='salmon')
bars2 = axes[0].bar(x + width/2, val_loss, width, label='Validation Loss', color='skyblue')

axes[0].set_ylabel('Log Loss', fontsize=11)
axes[0].set_title('Training vs Validation Loss', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Overfitting Gap
overfit_gaps = [xgb_base_overfit_gap, xgb_tuned_overfit_gap]
colors = ['red' if gap > 0.1 else 'green' for gap in overfit_gaps]
bars = axes[1].bar(models, overfit_gaps, color=colors, alpha=0.7)
axes[1].set_ylabel('Overfitting Gap (Train Loss - Val Loss)', fontsize=11)
axes[1].set_title('Overfitting Analysis', fontweight='bold')
axes[1].axhline(y=0.1, color='red', linestyle='--', label='Warning Threshold', linewidth=2)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# 6. Sensitivity vs Specificity
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Baseline', 'Tuned']
sensitivity = [xgb_base_sensitivity, xgb_tuned_sensitivity]
specificity = [xgb_base_specificity, xgb_tuned_specificity]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, sensitivity, width, label='Sensitivity (Recall)', color='orange')
bars2 = ax.bar(x + width/2, specificity, width, label='Specificity', color='purple')

ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Sensitivity vs Specificity Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('sensitivity_specificity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ All visualizations saved successfully!")