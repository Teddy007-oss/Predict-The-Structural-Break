import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import crunch

import sys
import os

# Add the repo root (one level up) to sys.path so "src" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your train and infer functions
#from src.train import train
#from src.infer import infer

# Load the Crunch Toolings
crunch = crunch.load_notebook()

# Step 1: Load parquet data
print("Loading data...")

#read the paquet files
y_train = pd.read_parquet(r"C:\Users\USER\Documents\ADIA LAB STRUCTURAL BREAK HACKATHON\data\y_train.parquet", engine='pyarrow')["structural_breakpoint"]
X_train = pd.read_parquet(r"C:\Users\USER\Documents\ADIA LAB STRUCTURAL BREAK HACKATHON\data\X_train.parquet", engine='pyarrow')
y_test = pd.read_parquet(r"C:\Users\USER\Documents\ADIA LAB STRUCTURAL BREAK HACKATHON\data\y_test.reduced.parquet", engine='pyarrow')["structural_breakpoint"]
X_test_df = pd.read_parquet(r"C:\Users\USER\Documents\ADIA LAB STRUCTURAL BREAK HACKATHON\data\X_test.reduced.parquet", engine='pyarrow')

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
