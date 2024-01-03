import re
import pandas as pd
from time_series_features import get_final_lab_df, create_lab_count_by_admit_time, create_lab_time_series_features
from utilities import IND_COLS, TRAIN_IND_COLS, get_columns_by_percentage, create_existence_features


def final_parsing_train(df, raw_lab_df, cat_attributes, cont_attributes, cont_cols_for_fillna, index_cols=TRAIN_IND_COLS, day_to_check=2):
    all_lab_df, lab_cont_attributes, cols_for_existence, stat_dict = get_final_lab_df(raw_lab_df, day_to_check=day_to_check)
    df = pd.merge(all_lab_df, df, on=index_cols, how='outer').sort_values('identifier')

    # removing drug columns from days we don't want
    cont_attributes = [col for col in cont_attributes if ("_days_" not in col or re.search(f"[{day_to_check}{day_to_check+2}]_days_", col) is not None)]
    cont_cols_for_fillna = [col for col in cont_cols_for_fillna if ("_days_" not in col or re.search(f"[{day_to_check}{day_to_check+2}]_days_", col) is not None)]

    # divides the count that was created before by time since admission
    cont_cols_for_fillna += create_lab_count_by_admit_time(df)
    n_cont_attributes = cont_attributes + lab_cont_attributes

    # we create existence features to cols with more than 5% non-null vals, but less than 30%
    cols_for_existence = get_columns_by_percentage(df, cols_for_existence, 0.05, 0.3)
    lab_cat_attributes = create_existence_features(df, cols_for_existence)
    n_cat_attributes = cat_attributes + lab_cat_attributes

    # filling "null values" of count features with zeros
    df[cont_cols_for_fillna].fillna(0, inplace=True)

    existing_is_imp = [col for col in df.columns if "_is_imputed" in col and col.replace("_is_imputed", '') in n_cont_attributes]
    df = df[list(index_cols) + sorted(n_cont_attributes + n_cat_attributes + existing_is_imp)].drop_duplicates()
    return df, n_cat_attributes, n_cont_attributes, cont_cols_for_fillna, cols_for_existence, stat_dict


def final_parsing_validation(df, raw_lab_df, train_cols_for_existence, train_cont_cols_for_fillna, train_cont_attributes, train_cat_attributes, stat_dict, ind_cols=IND_COLS, day_to_check=2):
    all_lab_df, _, _ = create_lab_time_series_features(raw_lab_df, "label", stat_dict, ind_cols, first_day=day_to_check)
    df = pd.merge(all_lab_df, df, on=ind_cols, how='outer').sort_values('identifier')

    # divides the count that was created before by time since admission
    create_lab_count_by_admit_time(df)

    # adding cols from train that are missing in validation
    missing_cols = list(set([col for col in train_cols_for_existence + train_cont_cols_for_fillna + train_cont_attributes + train_cat_attributes if col not in df.columns]))
    df = df.reindex(columns=df.columns.tolist() + missing_cols)

    # creating existence feature
    create_existence_features(df, train_cols_for_existence)

    # filling "null values" of count features with zeros
    df[train_cont_cols_for_fillna].fillna(0, inplace=True)

    train_cat_attributes = [col for col in train_cat_attributes if col != "IsolationForest"]
    df = df[list(ind_cols) + sorted(train_cont_attributes + train_cat_attributes)].drop_duplicates() # only taking features from train
    return df
