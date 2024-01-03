import pandas as pd
import scipy.cluster.hierarchy as hc
import numpy as np
from collections import defaultdict
from utilities import IND_COLS, get_org_col
import warnings
import scipy.spatial.distance as ssd
from feature_stats.get_feature_stats import get_cont_features_stats, get_categorical_features_stats

warnings.simplefilter("ignore")
N_KEEP_CAT = 1
PVAL_COL = 'pvalue_exponent'


def get_correlation_clusters(df, columns, correlation_threshold=0.85):
        corr_df = df[columns].corr(method='pearson').abs()
        corr_df[corr_df.isna()] = 0
        np.fill_diagonal(corr_df.values, 1)  # setting diagonal to 1 to deal with null cols
        ordered_cols = list(corr_df.columns)

        distances = 1 - corr_df.values
        distances = np.clip(distances, 0, 1, distances)
        distArray = ssd.squareform(distances)
        hier = hc.linkage(distArray, method="average")
        cluster_labels = hc.fcluster(hier, 1-correlation_threshold, criterion="distance")

        cluster_mapping = defaultdict(list)
        for ind in range(len(cluster_labels)):
            group_num = cluster_labels[ind]
            cluster_mapping[group_num].append(ordered_cols[ind])
        return list(cluster_mapping.values())


def get_features_to_drop(feature_list, p_df):
    rel_pvals = p_df[p_df['feature'].isin(feature_list)]
    if len(rel_pvals) == 0:
        print(f'No relevant pvals for {feature_list}')
        return feature_list
    best_feature = rel_pvals[rel_pvals[PVAL_COL] == max(rel_pvals[PVAL_COL])]['feature']
    if len(best_feature) > 0:
        best_feature = best_feature.iloc[0]
        feature_list.remove(best_feature)
    return feature_list


def filter_correlated_features_pval(pval_df, correlation_groups):
    to_drop = []
    for feature_list in correlation_groups:
        to_drop += get_features_to_drop(feature_list, pval_df)
    return to_drop


def filter_time_features_by_p_value(cols_to_check, pval_df, n_keep=2):
    features_to_keep = []
    cols_dict = defaultdict(list)

    for col in cols_to_check:
        cols_dict[get_org_col(col)].append(col)

    for feature_group in cols_dict.values():
        if len(feature_group) <= n_keep:
            features_to_keep += feature_group
        else:
            rel_pvals = pval_df[pval_df['feature'].isin(feature_group)]
            for i in range(n_keep):
                best_features = rel_pvals[rel_pvals[PVAL_COL] == rel_pvals[PVAL_COL].max()]['feature']
                best_feature = best_features.iloc[0] if len(best_features) > 0 else rel_pvals['feature'].iloc[0]
                features_to_keep.append(best_feature)
                rel_pvals = rel_pvals[rel_pvals['feature'] != best_feature]
    return features_to_keep


def filter_correlated_features(df, cont_stats_df, cat_stats_df, threshold, n_keep_cont, n_keep_cat=N_KEEP_CAT, ind_cols=IND_COLS):
    """removes correlated features using to clustering methods - Clustering by the original raw measurement
    and using hierarchical clustering on the features' correlation matrix.
    :param df: dataframe with the feature values.
    :param cont_stats_df: dataframe with stats about the continuous features, as returned by get_cont_features_stats in feature_stats/get_feature_stats
    :param cat_stats_df: dataframe with stats about the categorical features, as returned by get_categorical_features_stats in feature_stats/get_feature_stats
    :param threshold: correlation threshold. minimal correlation needed in order for two features to be included in the same cluster
    :param n_keep_cont: number of features to keep from the cluster of continuous features created from the same raw measurement.
    :param n_keep_cat: number of features to keep from the cluster of categorical features created from the same raw measurement. default: N_KEEP_CAT
    :param ind_cols: list of index columns
    :return list of all the features that passed the two filters
    """
    cont_stats_df = cont_stats_df.rename(columns={'t-test p-value_exponent': 'pvalue_exponent'})
    cat_stats_df = cat_stats_df.rename(columns={'chi2 p-value_exponent': 'pvalue_exponent'})
    pval_df = pd.concat([cont_stats_df, cat_stats_df])[['feature', 'pvalue_exponent']]

    filtered_cont_features = filter_time_features_by_p_value(cont_stats_df['feature'].tolist(), pval_df, n_keep=n_keep_cont)
    filtered_cat_features = filter_time_features_by_p_value(cat_stats_df['feature'].tolist(), pval_df, n_keep=n_keep_cat)
    filtered_features = filtered_cont_features + filtered_cat_features
    corr_groups = get_correlation_clusters(df, filtered_features, threshold)
    to_drop = filter_correlated_features_pval(pval_df, correlation_groups=corr_groups)
    filtered_features = [col for col in filtered_features if col not in to_drop]
    return sorted(filtered_features)


def filter_correlated_features_from_files(data_file, stat_files_prefix, threshold, n_keep_cont, n_keep_cat=N_KEEP_CAT, ind_cols=IND_COLS):
    """removes correlated features using to clustering methods - Clustering by the original raw measurement
    and using hierarchical clustering on the features' correlation matrix.
    :param data_file: path to csv with the feature values.
    :param stat_files_prefix: prefix to the feature stats files
    (as sent to get_categorical_features_stats and get_cont_features_stats in feature_stats/get_feature_stats)
    :param threshold: correlation threshold. minimal correlation needed in order for two features to be included in the same cluster
    :param n_keep_cont: number of features to keep from the cluster of continuous features
    created from the same raw measurement.
    :param n_keep_cat: number of features to keep from the cluster of categorical features
    created from the same raw measurement. default: N_KEEP_CAT
    :param ind_cols: list of index columns
    :return list of all the features that passed the two filters
    """
    df = pd.read_csv(data_file, index_col=ind_cols)
    cont_df = pd.read_csv(stat_files_prefix + "_cont_stats.csv")
    cat_df = pd.read_csv(stat_files_prefix + "_categorical_stats.csv")

    filtered_features = filter_correlated_features(df, cont_df, cat_df, threshold, n_keep_cont, n_keep_cat=n_keep_cat, ind_cols=IND_COLS)
    return filtered_features


def filter_correlated_features_from_stats(df, cont_features, cat_features, threshold, n_keep_cont, n_keep_cat=N_KEEP_CAT):
    """removes correlated features using to clustering methods - Clustering by the original raw measurement
    and using hierarchical clustering on the features' correlation matrix.
    :param df: dataframe with the feature values.
    :param cont_features: list of continuous features
    :param cat_features: list of categorical features
    :param threshold: correlation threshold. minimal correlation needed in order for two features to be included in the same cluster
    :param n_keep_cont: number of features to keep from the cluster of continuous features
    created from the same raw measurement.
    :param n_keep_cat: number of features to keep from the cluster of categorical features
    created from the same raw measurement. default: N_KEEP_CAT
    :return list of all the features that passed the two filters
    """
    cont_df, _ = get_cont_features_stats(df, cont_features, '', save_file=False)
    cat_df, _ = get_categorical_features_stats(df, cat_features, '', save_file=False)

    filtered_features = filter_correlated_features(df, cont_df, cat_df, threshold, n_keep_cont, n_keep_cat=n_keep_cat, ind_cols=IND_COLS)
    return filtered_features


