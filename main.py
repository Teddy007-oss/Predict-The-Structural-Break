import os
import joblib
import pandas as pd
import typing
import scipy.stats

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




def infer(X_test: typing.Union[pd.DataFrame, typing.Iterable[pd.DataFrame]],
          model_directory_path: str):
    """
    Inference function: applies selected test (t-test, KS-test, Mann–Whitney U).
    Works with parquet-loaded test data (either DataFrame or list of DataFrames).
    """
    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    method = model["method"]

    yield  # Mark as ready for Crunch runner

    # Case 1: X_test is one big DataFrame → split by id
    if isinstance(X_test, pd.DataFrame):
        grouped = (df for _, df in X_test.groupby(level="id"))
    else:
        # Case 2: Already a list of DataFrames
        grouped = X_test

    # Loop over each dataset
    for dataset in grouped:
        values_before = dataset.loc[dataset["period"] == 0, "value"]
        values_after = dataset.loc[dataset["period"] == 1, "value"]

        if method == "ttest":
            stat, p_value = scipy.stats.ttest_ind(values_before, values_after, equal_var=False)
        elif method == "ks":
            stat, p_value = scipy.stats.ks_2samp(values_before, values_after)
        elif method == "mannwhitney":
            stat, p_value = scipy.stats.mannwhitneyu(values_before, values_after, alternative="two-sided")
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert p-value into prediction score
        prediction = 1 - p_value
        print(f"[infer] id={id}, p-value={p_value:.4f}, prediction={prediction:.4f}")

        yield prediction
