from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
import json
from models.balanced_data_ensemble import BalancedDataEnsemble
from utilities import parse_col_name, split_train_test_by_indices, get_models_dict, IND_COLS, LABEL, ID_COL
from sklearn.feature_selection import SelectFromModel
import shap
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")
ensemble_model_count = 2
ENSEMBLE_SELECTION_METHODS = ['all_features', "0.75_features", '0.5_features']



def plot_shap(model, model_name, X_train, X_test, file_name):
    shap.initjs()
    explainer = shap.KernelExplainer(model, shap.sample(X_train, 5))
    test_shap_values = explainer.shap_values(X_test)
    n = len(X_test.columns)
    shap.summary_plot(test_shap_values, X_test, show=False, max_display=min(n, 30))
    title = f"{model_name} Shapley Values per Feature"

    plt.title(title)
    plt.savefig(file_name, dpi=350, bbox_inches='tight')
    plt.close()


def get_shap_best_features(model, X_train, is_tree=True):
    shap.initjs()
    if is_tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)[1]
    else:
        explainer = shap.KernelExplainer(model, shap.sample(X_train, 5))
        shap_values = explainer.shap_values(X_train)

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    return importance_df


def get_SelectFromModel_best_fetures(X, y, model):
    # selects all features with feature importance higher than mean feature importance
    sel = SelectFromModel(model)
    sel.fit(X, y.values.ravel())
    selected_feat = list(X.columns[(sel.get_support())])
    return selected_feat


def get_RFECV_best_features(X, y, model, nfold=5):
    print(X.count())
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(nfold, shuffle=True, random_state=42),
                  scoring='roc_auc_ovo_weighted')
    rfecv.fit(X, y.values.ravel())
    print(f"Optimal number of features according to RFECV: {rfecv.n_features_}")
    selected_feat = list(X.columns[(rfecv.get_support())])
    return selected_feat


def get_k_best_features(X, y, k=10, score_func=mutual_info_classif):
    # get list of best features
    attributes_arr = np.array(X.columns)
    best_features_mask = SelectKBest(score_func=score_func, k=k).fit(X, y).get_support()
    return list(attributes_arr[best_features_mask])


def get_ensemble_features(RFECV_features, SelectFromModel_features, k_shap_features, select_k_features):
    all_features = sorted(list(set(RFECV_features + SelectFromModel_features + k_shap_features + select_k_features)))
    feature_counter = Counter(RFECV_features + SelectFromModel_features + k_shap_features + select_k_features)
    common_features_75 = [k for k, v in feature_counter.items() if v / 4 >= 0.75]
    common_features_50 = [k for k, v in feature_counter.items() if v / 4 >= 0.5]
    return all_features, common_features_75, common_features_50


def get_feature_selection_dict(data_file, cols, train_ids, id_col=ID_COL, ind_cols=IND_COLS, label_col=LABEL, models=(), k=20, param_dict=None, res_json=""):
    """Creates a feature selection dict of the models, and saves it to the res_json"""
    df = pd.read_csv(data_file)
    if list(train_ids):  # we have folds
        X_train, y_train, _, _ = split_train_test_by_indices(df, train_ids, cols, id_col, ind_cols, label_col)
    else:
        df.set_index(ind_cols, inplace=True)
        X, y = df[cols], df[label_col]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.rename(columns={col: parse_col_name(col) for col in X_train.columns})
    k = k if k <= len(X_train.columns) else len(X_train.columns)
    models_dict = get_models_dict(models, param_dict, default_iter_num=len(df))
    models_features_dict = {}

    if "KNN" in models_dict:
        models_dict["KNN"].fit(X_train, y_train)
    for model_name, model in models_dict.items():
        is_tree = model_name == "Random Forest"
        if model_name == "KNN":  # KNN doesn't support this
            RFECV_features, SelectFromModel_features = [], []
        else:
            RFECV_features = get_RFECV_best_features(X_train, y_train, model)
            SelectFromModel_features = get_SelectFromModel_best_fetures(X_train, y_train, model)
        select_k_features = get_k_best_features(X_train, y_train, k)

        model.fit(X_train, y_train)
        rel_model = model if is_tree else lambda w: model.predict(w)
        shap_features = get_shap_best_features(rel_model, X_train, is_tree)
        k_shap_features = list(shap_features.iloc[:k]["column_name"])

        all_features, common_features_75, common_features_50 = get_ensemble_features(RFECV_features, SelectFromModel_features, k_shap_features, select_k_features)
        models_features_dict[model_name] = {"all_features": all_features, "RFECV_features": RFECV_features,
                                            "select_k_best": select_k_features, "shap_features": k_shap_features,
                                            "SelectFromModel_features": SelectFromModel_features,
                                            "0.75_features": common_features_75, "0.5_features": common_features_50}

    with open(res_json, "w") as f_json:
        json.dump(models_features_dict, f_json)

    return models_features_dict


def get_feature_selection(X_train, y_train, model_name, features_choice, K=20, param_dict=None):
    """Runs the feature selection method chosen"""
    K = K if K <= len(X_train.columns) else len(X_train.columns)
    model = get_models_dict([model_name], param_dict, default_iter_num=len(X_train))[model_name]
    RFECV_features, SelectFromModel_features, select_k_features, k_shap_features = [], [], [], []

    if model_name == "KNN":
        model.fit(X_train, y_train)
    else:  # KNN doesn't support them
        if features_choice in ENSEMBLE_SELECTION_METHODS + ['RFECV_features']:
            RFECV_features = get_RFECV_best_features(X_train, y_train, model)
        if features_choice in ENSEMBLE_SELECTION_METHODS + ['SelectFromModel_features']:
            SelectFromModel_features = get_SelectFromModel_best_fetures(X_train, y_train, model)

    if features_choice in ENSEMBLE_SELECTION_METHODS + ['select_k_best']:
        select_k_features = get_k_best_features(X_train, y_train, K)

    model.fit(X_train, y_train)
    if features_choice in ENSEMBLE_SELECTION_METHODS + ['shap_features']:
        is_tree = model_name == "Random Forest"
        rel_model = model if is_tree else lambda w: model.predict(w)
        shap_features = get_shap_best_features(rel_model, X_train, is_tree)
        k_shap_features = list(shap_features.iloc[:K]["column_name"])

    all_features, common_features_75, common_features_50 = get_ensemble_features(RFECV_features, SelectFromModel_features, k_shap_features, select_k_features)
    models_features_dict = {"all_features": all_features, "RFECV_features": RFECV_features, "select_k_best": select_k_features, "shap_features": k_shap_features, "SelectFromModel_features": SelectFromModel_features, "0.75_features": common_features_75, "0.5_features": common_features_50}
    return models_features_dict[features_choice]


def get_shap_figure_for_final_features(data_file, model_name, ind_cols=IND_COLS, label_col=LABEL, param_dict=None, feature_json="", features_selected="", validation_file="", output_file=""):
    is_data_ensemble = "DataEnsemble" in model_name
    raw_model_name = model_name.replace("DataEnsemble", "")
    df = pd.read_csv(data_file).set_index(ind_cols)
    v_df = pd.read_csv(validation_file)

    with open(feature_json, "r") as f_json:
        cols = json.load(f_json)[raw_model_name][features_selected]

    df = df.rename(columns={col: parse_col_name(col) for col in df.columns})
    v_df = v_df.rename(columns={col: parse_col_name(col) for col in v_df.columns})

    X_train, X_test, y_train, y_test = df[cols], v_df[cols], df[label_col], v_df[label_col]
    model_func = get_models_dict([raw_model_name], param_dict, default_iter_num=len(df), get_funcs=True)[raw_model_name]
    model = BalancedDataEnsemble(model_func, model_count=ensemble_model_count) if is_data_ensemble else model_func()
    model.fit(X_train, y_train)

    plot_shap(model, model_name, X_train, X_test, output_file)

