from utilities import IND_COLS, TIME_LIMIT, TIME_COL, create_existance_features_per_col, unite_features


def get_procedure_df(pr_df, index_cols=IND_COLS):
    # filtering out entries of procedures that took place after our prediction time
    pr_df = pr_df[pr_df[TIME_COL] >= -TIME_LIMIT].drop(columns=[TIME_COL])
    unite_features(pr_df, "label")
    pr_df, procedures_attributes = create_existance_features_per_col(pr_df, ["label"], index_cols)

    return pr_df, procedures_attributes
