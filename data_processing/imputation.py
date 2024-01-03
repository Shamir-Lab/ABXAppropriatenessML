"""Data imputation - KNN Imputation using different distance methods"""
import numpy as np
from scipy.special import comb
from sklearn.impute import KNNImputer, SimpleImputer


class PenaltyImputer:
    def __init__(self, ratio, k):
        self.k = k
        self.ratio = ratio
        self.imp = None # will be initialized with the fit function.
        self.penalties = [] # will be initialized with the fit function.

    def __get_null_penalty_per_col(self, col):
        n = max(2, round(self.ratio * col.count()))
        denominator = comb(n, 2)  # n choose 2
        step = 1 / n
        samples = np.array([np.nanquantile(col, min([q, 1])) for q in np.arange(step, 1 + step, step)])
        penalty = sum([sum((i - samples[np.where(samples > i)]) ** 2) for i in samples[:-1]]) / denominator
        return penalty

    def __knn_dist(self, X, Y, missing_values=np.nan, **kwds):
        vec = (X - Y) ** 2
        vec[np.where(np.isnan(vec))] = self.penalties[np.where(np.isnan(vec))]
        score = np.sqrt(np.sum(vec))
        return score

    def fit(self, X, y=None):
        self.penalties = X.apply(self.__get_null_penalty_per_col, axis=0).to_numpy()
        self.imp = KNNImputer(missing_values=np.nan, n_neighbors=self.k, weights='distance', metric=self.__knn_dist)
        return self.imp.fit(X)

    def transform(self, X):
        return self.imp.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


class WeightedImputer:
    def __init__(self, k):
        self.imp = KNNImputer(missing_values=np.nan, n_neighbors=k, weights='distance',
                         metric=self.__weighted_distance)

    def __weighted_distance(self, X, Y, missing_values=np.nan, **kwds):
        vec = (X - Y) ** 2
        count = np.count_nonzero(~np.isnan(vec))
        score = np.sqrt(np.nansum(vec)) / count
        return score

    def fit(self, X, y=None):
        return self.imp.fit(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.imp.fit_transform(X,  y, **fit_params)

    def transform(self, X):
        return self.imp.transform(X)


def get_imputer(imp, k, ratio=0.1):
    if imp == 'default':
        return KNNImputer(missing_values=np.nan, n_neighbors=k, weights='distance')
    elif imp == 'penalty':
        return PenaltyImputer(ratio, k)
    elif imp == 'weighted':
        return WeightedImputer(k)
    else:
        return SimpleImputer(missing_values=np.nan, strategy=imp)


def fill_cat_null_values(df, cols_for_na, unknown_ethnicity="is_Other/Unknown"):
    if unknown_ethnicity in cols_for_na:
        df[unknown_ethnicity] = df[unknown_ethnicity].fillna(1)
    cols_to_add = [col for col in cols_for_na if col not in df.columns]
    df = df.reindex(columns=df.columns.tolist() + cols_to_add, fill_value=0)
    df[cols_for_na] = df[cols_for_na].fillna(0)
    return df