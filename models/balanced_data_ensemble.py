import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTENC


class BalancedDataEnsemble:
    def __init__(self, model_func, model_count=2):
        self.model_count = model_count
        self.models = [model_func() for i in range(model_count)]

    def fit(self, X, y):
        y_pos, y_neg = y[y == 1], y[y == 0]
        n = len(y_neg) // self.model_count
        mod = len(y_neg) % self.model_count
        shuffled_idx = np.random.RandomState(seed=42).permutation(y_neg.index)
        curr_index = 0
        for i in range(self.model_count):
            final_indice = n+1 if mod > 0 else n
            mod -= 1
            rel_indices = np.random.RandomState(seed=42).permutation(list(shuffled_idx[curr_index:curr_index+final_indice]) + list(y_pos.index))
            if isinstance(y.index, pd.MultiIndex):
                rel_indices = [tuple([item[0], int(item[1]), int(item[2])]) for item in rel_indices]
            curr_index = final_indice
            rel_X, rel_y = X.loc[rel_indices, :], y[rel_indices]
            self.models[i].fit(rel_X, rel_y)
        return self

    def predict_proba(self, X):
        probs = pd.DataFrame([clf.predict_proba(X)[:, 1] for clf in self.models]).mean(axis=0)
        return np.array([1 - probs, probs]).T


    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return np.array([int(p >= 0.5) for p in probs])

    def score(self, X_test, y_test):
        return np.median([clf.score(X_test, y_test) for clf in self.models])


def balance_data(X, y, method, ratio, cat_cols=()):
    if ratio is None:  # no balancing needed
        return X, y
    sampling_strategy = ratio / (1 - ratio)
    if method == "ADASYN":
        X_balanced, y_balanced = ADASYN(sampling_strategy=sampling_strategy, random_state=42).fit_resample(X, y)
    elif method == "BorderlineSMOTE":
        X_balanced, y_balanced = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42).fit_resample(X,y)
    elif method == "SMOTENC":
        curr_cat_features = [col for col in cat_cols if col in X.columns]
        cat_indices = X.columns.get_indexer(curr_cat_features)
        X_balanced, y_balanced = SMOTENC(cat_indices, sampling_strategy=sampling_strategy, random_state=42).fit_resample(X, y)
    else:  # not supported
        X_balanced, y_balanced = X, y
    return  X_balanced, y_balanced

