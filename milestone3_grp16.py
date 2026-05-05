import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# DEBUG: CHECK CURRENT DIRECTORY
# ==========================================
print("Current working directory:", os.getcwd())
print("Files in this folder:", os.listdir())

# ==========================================
# DATA INGESTION
# ==========================================
print("\nLoading Bank Marketing Dataset...")

# 🔥 TEMP FIX: Use FULL PATH (guaranteed to work)
df = pd.read_csv(r"C:\Users\Tanusha Suresh\Documents\Python\bank-full.csv", sep=";")

print("Original dataset shape:", df.shape)

# ==========================================
# CLEANING & PREPROCESSING
# ==========================================
df = df.drop_duplicates()

if "duration" in df.columns:
    df = df.drop("duration", axis=1)

df["y"] = df["y"].map({"no": 0, "yes": 1})

X = df.drop("y", axis=1)
y = df["y"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

print("Dataset shape after preprocessing:", X.shape)

# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nData Split Summary")
print("------------------")
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])

# ==========================================
# MODEL SETUP
# ==========================================
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ==========================================
# TRAINING LOOP
# ==========================================
print("\nStarting XGBoost Training Loop...")

start_time = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# ==========================================
# METRICS
# ==========================================
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("\nValidation Metrics")
print("------------------")
print("Accuracy  :", round(accuracy_score(y_val, y_pred), 4))
print("Precision :", round(precision_score(y_val, y_pred), 4))
print("Recall    :", round(recall_score(y_val, y_pred), 4))
print("F1 Score  :", round(f1_score(y_val, y_pred), 4))
print("ROC-AUC   :", round(roc_auc_score(y_val, y_prob), 4))

# ==========================================
# GRAPH
# ==========================================
results = model.evals_result()

train_loss = results["validation_0"]["logloss"]
val_loss = results["validation_1"]["logloss"]

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Log Loss")
plt.plot(val_loss, label="Validation Log Loss")
plt.title("XGBoost Training vs Validation Loss")
plt.xlabel("Boosting Round")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.show()