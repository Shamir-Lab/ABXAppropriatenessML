import re
import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier
import pandas as pd
import os

union_json = "double_names_mimic.json"
with open(os.path.join("feature_engineering", "utilities_files", union_json), "r") as f_json:
    mimic_union_dict = json.load(f_json)

IND_COLS = ("identifier", "subject_id", "hadm_id")
TRAIN_IND_COLS = ("identifier", "subject_id", "hadm_id", "target")
ID_COL = "identifier"
LABEL = "target"
TIME_LIMIT = 24
TIME_COL = 'hours_from_charttime_time_to_targettime'
MIN_NON_EMPTY = 0.3
LOWER_MIN_NON_EMPTY = 0.05


def parse_col_name(name):
    name = re.sub("\s\([0-9]+\)", "", name.replace("_", " "))
    name = re.sub(r'[^A-Za-z0-9_]+', '', name.replace(" ", "_"))
    return name


def get_columns_by_percentage(df, columns, lower_percentage=LOWER_MIN_NON_EMPTY, upper_percentage=MIN_NON_EMPTY):
    new_columns = [col for col in columns if lower_percentage <= df[col].count()/len(df) < upper_percentage]
    return new_columns


def create_existence_features(df, columns):
    new_cols = []
    for col_name in columns:
        new_col_name = f"{col_name}_existence"
        if col_name not in df.columns:
            df[new_col_name] = 0
        else:
            df[new_col_name] = (~df[col_name].isna()).replace({True: 1, False: 0})
        new_cols.append(new_col_name)
    return new_cols


def unite_features(df, col):
    df[col] = df[col].apply(lambda x: mimic_union_dict[x] if x in mimic_union_dict else x)


def filter_cols_by_null_percentage(df, columns, percentage):
    new_columns = [col for col in columns if df[col].count()/len(df) >= percentage]
    return new_columns


def get_org_col(col):
    """Return the name of the raw measurement that the col was created from"""
    if "_days_" in col:
        return re.search(".+(?=_all_days_)", col).group() if "_all_days" in col else re.search(".+(?=_[2345]_days_)", col).group()
    if "_over_time" in col:
        return re.search("(?<=count_).+(?=_over_time)", col).group()
    else:
        return col.replace("_unique_percentage", "").replace("time_from_last_","").replace("_12_hours", "").replace("_median_diff", "").replace("_coef_ratio", "").replace("_is_imputed", "").replace("_after_before_BC_ratio", "")


def get_models_dict(models, param_dict, default_iter_num=100, get_funcs=False):
    param_dict = {} if param_dict is None else param_dict
    rf = (lambda: RandomForestClassifier(random_state=42, **param_dict["Random Forest"])) if "Random Forest" in param_dict else (lambda: RandomForestClassifier(random_state=42))
    knn = (lambda: KNeighborsClassifier(**param_dict["KNN"])) if "KNN" in param_dict else (lambda: KNeighborsClassifier())
    adb = (lambda: AdaBoostClassifier(random_state=42, **param_dict["Adaboost"])) if "Adaboost" in param_dict else (lambda: AdaBoostClassifier(random_state=42))
    lr = (lambda: LogisticRegression(random_state=42, **param_dict["Logistic Regression"])) if "Logistic Regression" in param_dict else (lambda: LogisticRegression(random_state=42, max_iter=default_iter_num))
    svm = (lambda: SVC(probability=True, random_state=42, kernel='linear', **param_dict["SVM"])) if "SVM" in param_dict else (lambda: SVC(probability=True, random_state=42, kernel='linear'))
    sgd = (lambda: SGDClassifier(**param_dict["SGD"], random_state=42)) if "SGD" in param_dict else (lambda: SGDClassifier(loss="log", random_state=42))
    xgb_model = (lambda: xgb.XGBClassifier(objective="binary:logistic", eval_metric=['auc', 'aucpr'], random_state=42, booster='dart', rate_drop=0.1, **param_dict["xgboost"])) if "xgboost" in param_dict else (lambda: xgb.XGBClassifier(objective="binary:logistic", eval_metric=['auc', 'aucpr'], booster='dart', rate_drop=0.1, random_state=42))
    grad = (lambda: GradientBoostingClassifier(random_state=42, **param_dict["GradientBoosting"])) if "GradientBoosting" in param_dict else (lambda: GradientBoostingClassifier(random_state=42))
    lgbm = (lambda: LGBMClassifier(random_state=42, **param_dict["LightGBM"])) if "LightGBM" in param_dict else (lambda: LGBMClassifier(random_state=42))

    models_dict = {"Random Forest": rf, "KNN": knn, "Adaboost": adb, "Logistic Regression": lr, "SVM": svm, "SGD":sgd,
                   "xgboost": xgb_model, "GradientBoosting":grad, "LightGBM": lgbm}
    models_dict = {key: v for (key, v) in models_dict.items() if key in models} if models else models_dict

    if not get_funcs:
        models_dict = {key: v() for key, v in models_dict.items()}
    return models_dict


def split_train_test_by_indices(df, train_ids, cols, id_col=ID_COL, ind_cols=IND_COLS, label_col=LABEL):
    train = df[df[id_col].isin(train_ids)].set_index(ind_cols)
    test = df[~df[id_col].isin(train_ids)].set_index(ind_cols)
    X_train, y_train = train[cols], train[label_col]
    X_test, y_test = test[cols], test[label_col]
    return X_train, y_train, X_test, y_test


def create_existance_features_per_col(df, cols_to_parse, index_cols=IND_COLS):
    for col_to_parse in cols_to_parse:
        feature_types = set(df[col_to_parse])
        for feature_type in feature_types:
            if pd.notnull(feature_type):
                col_name = feature_type + "_existence"
                df[col_name] = df[col_to_parse].apply(lambda x: 1 if x == feature_type else 0)

    if 'spec_type_desc' in cols_to_parse:
        total_count = df.groupby(index_cols)['spec_type_desc'].count().rename("Total_Culture")
        df = df.set_index(index_cols).join(total_count, how='outer').reset_index()
    df.drop(columns=cols_to_parse, inplace=True)
    attributes = [col for col in df.columns if col not in index_cols]
    c_df = df.groupby(index_cols).max().reset_index().drop_duplicates()
    return c_df, attributes


def get_is_imputed_cols(df, columns):
    imp_cols = [f"{col}_is_imputed" for col in columns]
    missing_cols = [col for col in imp_cols if col not in df.columns]
    df = df.reindex(columns=df.columns.tolist()+missing_cols)
    for col in missing_cols:
        df[col] = df[col.replace("_is_imputed", "")].isnull()
    df[missing_cols] = df[missing_cols].astype('int8')
    df[imp_cols] = df[imp_cols].fillna(1)
    return df, imp_cols

