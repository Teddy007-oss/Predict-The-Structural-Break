import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import crunch

# Import your train and infer functions
from src.train import train
from src.infer import infer

# Load the Crunch Toolings
crunch = crunch.load_notebook()

# Step 1: Load parquet data
print("Loading data...")

X_train = pd.read_parquet("data/x_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")["structural_breakpoint"]

X_test_df = pd.read_parquet("data/x_test_reduced.parquet")
y_test = pd.read_parquet("data/y_test_reduced.parquet")["structural_breakpoint"]

# IMPORTANT: X_test must be a list of DataFrames (one per id)
X_test = [df for _, df in X_test_df.groupby("id")]

print(f"Train shape: {X_train.shape}")
print(f"Test sets: {len(X_test)}")


# ----------------------------
# Step 2: Run crunch.test()
# ----------------------------
print("\nRunning crunch local test...")
crunch.test(
    # Uncomment to disable re-training every time
    # force_first_train=False,
    
    # Uncomment if determinism check causes issues
    # no_determinism_check=True,
)


# ----------------------------
# Step 3: Inspect predictions
# ----------------------------
print("\nLoading predictions...")
prediction = pd.read_parquet("data/prediction.parquet")
print(prediction.head())

# ----------------------------
# Step 4: Local scoring
# ----------------------------
print("\nScoring locally...")
score = roc_auc_score(y_test, prediction)
print("Local ROC AUC:", score)
