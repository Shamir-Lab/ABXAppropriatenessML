from scipy.stats import ttest_ind
import pandas as pd
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
from math import log10


def apply_FDR_correction(df, column, alpha):
    relevant_values = df[pd.notnull(df[column])][column]
    _, df.loc[relevant_values.index, column] = fdrcorrection(relevant_values, alpha)
    df[column+"_exponent"] = df[column].apply(lambda x: np.nan if pd.isnull(x) else round(-log10(x), 2))
    df[column] = np.round(df[column], decimals=2)


def get_cont_features_stats(df, cont_features, file_prefix, save_file=True):
    """Apply t-test on the categorical features, applies FDR correction and saves it to a csv file if save_file == True.
     :param df: dataframe with the target and the relevant features and their values.
     :param cont_features: list of continuous features.
     :param file_prefix: prefix to output csv file. should be the same as the one sent to get_categorical_features_stats.
     :param save_file: whether to save the stats dataframe as a csv.
     :return stats dataframe anf list of all significant continuous features (alpha = 0.05) after FDR correction
     """
    stats = pd.DataFrame(cont_features, columns=["feature"])
    positive, negative = df[df["target"] == 1][cont_features], df[df["target"] == 0][cont_features]
    _, stats["t-test p-value"] = ttest_ind(positive, negative, equal_var=False, nan_policy='omit')
    apply_FDR_correction(stats, "t-test p-value", 0.025)
    stats = stats.round(2)
    if save_file:
        stats.to_csv(file_prefix + "_cont_stats.csv", index=False)
    return stats[['feature', 't-test p-value', 't-test p-value_exponent']], list(stats[stats['t-test p-value'] < 0.05]['feature'])


def get_categorical_features_stats(df, cat_features, file_prefix, save_file=True):
    """Apply Chi Squared test on the categorical features, applies FDR correction and saves it to a csv file if save_file == True.
    :param df: dataframe with the target and the relevant features and their values.
    :param cat_features: list of categorical features.
    :param file_prefix: prefix to output csv file. should be the same as the one sent to get_cont_features_stats.
    :param save_file: whether to save the stats dataframe as a csv.
    :return stats dataframe and list of all significant categorical features (alpha = 0.05) after FDR correction
    """
    rel_df, y = df[cat_features], df['target']
    # getting only columns with all positive values, as chi test only works with positive values
    positive_df = rel_df.loc[:, rel_df.ge(0).all()]
    positive_columns = list(positive_df.columns)
    chi2_p_values = chi2(positive_df, y)[1]

    result_dict = {feat: {"chi2 p-value": np.nan} for feat in cat_features}
    result_dict.update({positive_columns[i]: {"chi2 p-value": chi2_p_values[i]} for i in range(len(positive_columns))})
    stats = pd.DataFrame.from_dict(result_dict, orient='index').reset_index().rename(columns={"index": "feature"})
    apply_FDR_correction(stats, "chi2 p-value", 0.05)
    stats = stats.set_index('feature').loc[cat_features].reset_index()  # making it the same order
    stats = stats.round(2)
    if save_file:
        stats.to_csv(file_prefix + "_categorical_stats.csv", index=False)
    return stats, list(stats[stats["chi2 p-value"] < 0.05]['feature'])
