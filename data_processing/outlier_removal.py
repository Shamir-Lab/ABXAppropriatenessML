import numpy as np
from math import sqrt
import json


with open("human_range_dict.json", "r") as fin:
    HUMAN_DICT = json.load(fin)


def remove_inhuman_values(df, relevant_col, label):
    if label in HUMAN_DICT:
        min_val, max_val = HUMAN_DICT[label]
        df[relevant_col] = df[relevant_col].apply(lambda x: x if min_val <= x <= max_val else np.nan)
    return label in HUMAN_DICT


def remove_extreme_vals_IQR(df, label, relevant_cols, stat_dict=None, factor=1.5):
    if stat_dict is not None:
        for col in relevant_cols:
            if label not in stat_dict:
                df[col] = df[col].apply(lambda x: np.nan)  # remove all values as it's irrelevant
            else:
                q_25, q_75 = stat_dict[label]["q_25"], stat_dict[label]["q_75"]
                q_diff = stat_dict[label]["q_diff"]
                df[col] = df[col].apply(lambda x: x if q_25 - q_diff < x < q_75 + q_diff else np.nan)
    else:
        stat_dict = {}
        for col in relevant_cols:
            q_25 = np.nanquantile(df[col], 0.25)
            q_75 = np.nanquantile(df[col], 0.75)
            q_diff = factor*(q_75 - q_25)
            stat_dict[label] = {"q_25": q_25, "q_75": q_75, "q_diff": q_diff}
            df[col] = df[col].apply(lambda x: x if q_25 - q_diff < x < q_75 + q_diff else np.nan)

    return stat_dict


def remove_extreme_vals_z_score(df, label, rel_cols, stat_dict=None, sd_num=3, factor=1/sqrt(2)):
    if stat_dict is not None:
        for col in rel_cols:
            if label not in stat_dict:
                df[col] = df[col].apply(lambda x: np.nan)  # remove all values as it's irrelevant
            else:
                mean = stat_dict[label]["mean"]
                sd = stat_dict[label]["std"]
                df[col] = df[col].apply(lambda x: x if mean - sd < x < mean + sd else np.nan)

    else:
        stat_dict = {}
        for col in rel_cols:
            mean = df[col].mean()
            sd = factor * df[col].std() * sd_num
            stat_dict[label] = {"mean": mean, "std": sd}
            df[col] = df[col].apply(lambda x: x if mean - sd < x < mean + sd else np.nan)

    return stat_dict


def remove_extreme_values(df, relevant_col, label, stat_dict=None, is_z_score=True):
    removed_inhuman = remove_inhuman_values(df, relevant_col, label)
    if is_z_score:
        stat_dict = remove_extreme_vals_z_score(df, label, [relevant_col], stat_dict)
    else:
        stat_dict = remove_extreme_vals_IQR(df, label, [relevant_col], stat_dict)

    stat_dict = stat_dict if stat_dict is not None else {}
    return stat_dict
