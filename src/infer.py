import typing
import joblib
import scipy.stats

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

        yield prediction
