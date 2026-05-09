from sklearn.model_selection import GridSearchCV
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

# Define parameter grid (less extreme)
param_grid = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.05, 0.1, 1.5],
    'subsample': [0.8, 0.85, 0.9],           # Softer range
    'colsample_bytree': [0.8, 0.85, 0.9],    # Softer range
    'reg_alpha': [0.01, 0.05, 0.1],             # Less aggressive
    'reg_lambda': [0.5, 1.0, 1.5],           # Moderate range
    'max_depth': [3, 4, 5],                  # Broader depth range
}

xgb_base = XGBClassifier(
   # n_estimators=500,
   # learning_rate=0.05,
    random_state=SEED,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=5,                        # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)
print(f"Best Cross-Val Accuracy: {grid_search.best_score_:.4f}")

# Train final model with best parameters
xgb_optimal = grid_search.best_estimator_
xgb_optimal_acc = accuracy_score(y_val, xgb_optimal.predict(X_val))
print(f"Optimal Model Validation Accuracy: {xgb_optimal_acc:.4f}")