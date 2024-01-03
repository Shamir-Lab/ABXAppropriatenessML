from utilities import get_models_dict, filter_cols_by_null_percentage, get_is_imputed_cols
from balanced_data_ensemble import BalancedDataEnsemble, balance_data
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_processing.imputation import get_imputer, fill_cat_null_values
from feature_selection.feature_selection import get_feature_selection
from feature_selection.correlation_filteration import filter_correlated_features_from_stats
import pandas as pd
import numpy as np


class ABXModel:
    """
    The ABXModel object represents a general ML model for predicting ABX appropriateness, capturing the pre-training steps.
    """
    def __init__(self, model_name, features_choice, K, param_dict, imp, norm, cont_cols, cat_cols, k=5, data_ensemble=False, null_thresh=0.7, corr_threshold=0.7, n_keep=1, balancing="BorderlineSMOTE", balancing_ratio=None, get_imp_cols=True):
        # Pre-processing parameters
        self.get_imp_cols = get_imp_cols
        self.features_choice = features_choice
        self.K = K
        self.corr_threshold = corr_threshold
        self.n_keep = n_keep
        self.normalizer = MinMaxScaler() if norm == 'min_max' else StandardScaler()
        self.imputer = get_imputer(imp, k)
        self.balancing = balancing
        self.balancing_ratio = balancing_ratio
        self.isolation_forest = IsolationForest(random_state=42)
        self.null_threshold = null_thresh

        # Model parameters
        self.model_name = model_name
        self.param_dict = param_dict
        self.clf = get_models_dict([model_name], param_dict, get_funcs=True)[model_name]
        self.clf = BalancedDataEnsemble(self.clf) if data_ensemble else self.clf()
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.selected_features = []

    def fit(self, X_train, y_train):
        """ Train model """
        print(f"Train {self.model_name}.\nTraining set size: {len(X_train)}")

        cat_cols = [col for col in self.cat_cols if col in X_train.columns]
        cont_cols = [col for col in self.cont_cols if col in X_train.columns]

        # filtering Features
        cont_cols = filter_cols_by_null_percentage(X_train, cont_cols, self.null_threshold)
        var_threshold = VarianceThreshold(threshold=0.005)
        var_threshold.fit(X_train[cont_cols])
        cont_cols = X_train[cont_cols].loc[:, var_threshold.get_support()].columns.tolist()

        if self.get_imp_cols:  # create is_imputed features
            X_train, imp_cols = get_is_imputed_cols(X_train, cont_cols)
            cat_cols += list(np.array(imp_cols)[np.where(X_train[imp_cols].sum() > 0)[0]])  # only columns that were imputes

        X_train = X_train[sorted(cont_cols + cat_cols)]
        X_train.columns = X_train.columns.astype(str)
        self.cont_cols, self.cat_cols = cont_cols, cat_cols  # Saving for evaluation

        # Standardization
        X_train[self.cont_cols] = self.normalizer.fit_transform(X_train[self.cont_cols])

        # Data imputation
        X_train = fill_cat_null_values(X_train, self.cat_cols)
        X_train[self.cont_cols] = self.imputer.fit_transform(X_train[self.cont_cols])

        # Getting Isolation Forest Feature (anomaly score)
        self.isolation_forest.fit(X_train)
        X_train.loc[:, 'IsolationForest'] = self.isolation_forest.predict(X_train)
        X_train["IsolationForest"] = X_train["IsolationForest"].map({1: 1, -1: 0})
        self.cat_cols.append('IsolationForest')

        # Redundant features removal + feature selection
        filtered_cols = filter_correlated_features_from_stats(pd.concat([X_train, y_train], axis=1), self.cont_cols, self.cat_cols, self.corr_threshold, self.n_keep)
        self.selected_features = get_feature_selection(X_train[filtered_cols], y_train, self.model_name, self.features_choice, self.K, {})

        # balance data
        rel_cat_cols = [col for col in X_train.columns if col in self.cat_cols]
        balanced_X_train, balanced_y_train = balance_data(X_train, y_train, self.balancing, self.balancing_ratio, cat_cols=rel_cat_cols)

        # Train model
        self.clf.fit(balanced_X_train[self.selected_features], balanced_y_train)

    def evaluation(self, X_test, y_test):
        """ Predict and evaluate model """
        print(f"Evaluate {self.model_name}.\nTest set size: {len(X_test)}")

        # adding cols from train that are missing in test
        if self.get_imp_cols:
            X_test, _ = get_is_imputed_cols(X_test, [col.replace("_is_imputed", "") for col in self.cat_cols if "_is_imputed" in col])  # adding mask cols

        rel_cat_cols = [col for col in self.cat_cols if col != 'IsolationForest']
        missing_cols = list(set([col for col in self.cont_cols + rel_cat_cols if col not in X_test.columns]))
        X_test = X_test.reindex(columns=X_test.columns.tolist() + missing_cols)
        X_test = X_test[sorted(rel_cat_cols + self.cont_cols)]
        X_test.columns = X_test.columns.astype(str)

        # Standardization
        X_test[self.cont_cols] = self.normalizer.transform(X_test[self.cont_cols])

        # Data imputation
        X_test = fill_cat_null_values(X_test, rel_cat_cols)
        X_test[self.cont_cols] = self.imputer.transform(X_test[self.cont_cols])

        # Getting Isolation Forest Feature (anomaly score)
        X_test.loc[:, 'IsolationForest'] = self.isolation_forest.predict(X_test)
        X_test["IsolationForest"] = X_test["IsolationForest"].map({1: 1, -1: 0})

        # Feature selection
        X_test = X_test[self.selected_features]

        # Predict
        probs = self.clf.predict_proba(X_test)[:, 1]
        model_results = {'score': probs, 'target': y_test}
        res_df = pd.DataFrame.from_dict(model_results)
        return res_df