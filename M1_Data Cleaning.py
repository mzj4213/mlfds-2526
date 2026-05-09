# Milestone 1 - Data Pipeline
# Bank Marketing Dataset


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split


# Preliminary Setup
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# 1. Loading the dataset
df = pd.read_csv("dataset.csv", sep=';')
print("Initial loading shape: ", df.shape)


# 2. Cleaning
# Remove of 'duration' column as it brings unneccessary bias
df = df.drop(columns='duration', axis=1)


# Only accept success for 'poutcome' to find relationship between previous and current campaign
df['poutcome'] = (df['poutcome'] == 'success')
# Converts boolean column to 1/0
df['poutcome'] = df['poutcome'].astype(int)


# Replace 'unknown' with NA for remaining columns to handle imputation
df.replace("unknown", pd.NA, inplace=True)
# Impute missing values: Mode for categorical; Mean for numerical
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    elif df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())


# Drop duplicates
df = df.drop_duplicates()


# Convert 'yes/no' to 1/0 for binary columns
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})


print("Post cleaning shape: ", df.shape)


# 3. Splitting the dataset
# Split X and y
X = df.drop('y', axis=1)
y = df['y']


# Train/Validation/Test Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED
)


print(f"Train size:      {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test size:       {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")


print("Pre encoding preview:")
print(df.head())


# 4. Encoding
# A. Ordinal Encoding (Education has a logical order)
edu_order = ['primary', 'secondary', 'tertiary']
ord_enc = OrdinalEncoder(categories=[edu_order], handle_unknown='use_encoded_value', unknown_value=-1)


# B. One-Hot Encoding (Nominal variables like job, marital, etc.)
ohe_cols = ['job', 'marital', 'contact', 'month']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# C. Scaling (Numerical variables)
num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()


# --- APPLYING TO TRAIN (Fit and Transform) ---
# Fit and Transform Train
X_train_num = scaler.fit_transform(X_train[num_cols])
X_train_ord = ord_enc.fit_transform(X_train[['education']])
X_train_ohe = ohe.fit_transform(X_train[ohe_cols])
# Convert boolean poutcome to 1/0 for math compatibility
X_train_pout = X_train[[ 'default', 'housing', 'loan','poutcome']].astype(int).values


# Combine back into a single array/dataframe for training
X_train_final = np.hstack([X_train_num, X_train_ord, X_train_ohe, X_train_pout])


# --- APPLYING TO VAL & TEST (Transform ONLY) ---
def transform_data(data):
    ord_part = ord_enc.transform(data[['education']])
    ohe_part = ohe.transform(data[ohe_cols])
    num_part = scaler.transform(data[num_cols])
    pout_part = data[['default','housing','loan','poutcome']].astype(int).values
    return np.hstack([num_part, ord_part, ohe_part, pout_part])


X_val_final = transform_data(X_val)
X_test_final = transform_data(X_test)


print(f"Post encoding feature count: {X_train_final.shape[1]}")


# Summary Statistics
print("\nInfo:")
print(df.describe(include=['O']).T)


# Exploratory Data Analysis


# Distribution Plots
sns.set_theme(style="whitegrid")
numerical_cols = ['age', 'balance', 'day', 'campaign', 'pdays']
plt.figure(figsize=(14, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')


plt.tight_layout()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 10))
# Calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)
# Create a heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()


# Bar chart showing number of clients that subscribed to a term deposit
df['y'].value_counts().plot(kind='bar')
plt.title("Number of clients that subscribed to a term deposit")
plt.show()


# =========================
# 💾 SAVE FINAL DATA (FIXED)
# =========================

print("Saving files...")

# 1. Save Features (X)
np.save("X_train.npy", X_train_final)
np.save("X_val.npy", X_val_final)
np.save("X_test.npy", X_test_final)

# 2. Save Targets (y)
np.save("y_train.npy", y_train.values if hasattr(y_train, 'values') else y_train)
np.save("y_val.npy", y_val.values if hasattr(y_val, 'values') else y_val)
np.save("y_test.npy", y_test.values if hasattr(y_test, 'values') else y_test)

# 3. Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("Done! 7 files are ready to be uploaded directly to GitHub.")

# 🔥 IMPORTANT: force matplotlib to release session
plt.close('all')

