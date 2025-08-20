import os
import joblib
import pandas as pd

def train(X_train: pd.DataFrame, y_train: pd.Series, model_directory_path: str, method: str = "ks"):
    """
    Train function for structural break detection.
    Since these are non-parametric/statistical tests, no actual training is done.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (time series data).
    y_train : pd.Series
        Training labels (structural break or not).
    model_directory_path : str
        Directory where the model will be saved.
    method : str
        Which test to use: 'ttest', 'ks', 'mannwhitney'.
    """
    # Save chosen method for inference
    model = {"method": method}
    
    joblib.dump(model, os.path.join(model_directory_path, "model.joblib"))
#hello