import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utilities import TRAIN_IND_COLS, TIME_COL, TIME_LIMIT, get_is_imputed_cols
from data_processing.outlier_removal import remove_extreme_values


def create_lab_count_by_admit_time(df):
    rel_cols = [col for col in df.columns if "_over_time" in col]
    for col_name in rel_cols:
        df[col_name] = df[col_name] / df["hours_from_admittime_to_targettime"]
    return rel_cols


def get_last_value(out_group, index_cols=TRAIN_IND_COLS):
    ggb = out_group.groupby(index_cols)
    last_val = out_group[out_group[TIME_COL] == ggb[TIME_COL].transform('min')].set_index(index_cols)
    last_val = last_val[~last_val.index.duplicated(keep='first')]  # keeping one value per person
    return last_val


def get_last_value_df(out_group, col_name, index_cols=TRAIN_IND_COLS):
    out_group = out_group[out_group[TIME_COL] >= -TIME_LIMIT]
    last_val = get_last_value(out_group, index_cols)
    last_val.rename(columns={'valuenum': col_name, TIME_COL: f"time_from_last_{col_name}_to_BC"}, inplace=True)
    last_val.drop(columns=['label'], inplace=True)

    if '_12_hours' in col_name:
        last_val.drop(f"time_from_last_{col_name}_to_BC", 1, inplace=True)
        return last_val
    else:
        before_BC = get_last_value(out_group[out_group[TIME_COL] > 0], index_cols)
        after_BC = get_last_value(out_group[out_group[TIME_COL] <= 0], index_cols)
        after_before_ratio = (after_BC['valuenum'] / before_BC['valuenum']).rename(f"{col_name}_after_before_BC_ratio")
        last_val = last_val.join(after_before_ratio)
        last_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    return last_val


def get_regression(X, y, num_limit=3):
    if len(y) < num_limit:
        return np.array([np.nan, np.nan, np.nan])

    else:
        X = -np.array(X).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        coef = reg.coef_[0]
        r2 = reg.score(X, y)
        return np.array([coef, r2, reg])


def calc_regression(X, y, num_limit=3):
    if len(y) < num_limit:
        return np.nan
    else:
        X = -np.array(X).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        return reg


def fill_null_by_reg_for_patient(row, reg_model_new):
    if row.name not in reg_model_new.index:
        return row
    reg_patient = reg_model_new.loc[row.name][0]
    if pd.isnull(reg_patient):
        return row
    coef = reg_patient.coef_[0]
    days_num = int(re.search(r'(?<=_)[2345](?=_days)', row.index[0])[0])
    row.iloc[0] = reg_patient.predict([[-(days_num * 24 - TIME_LIMIT) / 2]])[0]  # median
    if coef < 0:
        row.iloc[1] = reg_patient.predict([[-(days_num*24)]])[0]  # max
        row.iloc[2] = reg_patient.predict([[TIME_LIMIT]])[0]  # min
    else:
        row.iloc[1] = reg_patient.predict([[TIME_LIMIT]])[0]  # max
        row.iloc[2] = reg_patient.predict([[-(days_num*24)]])[0]  # min

    row.iloc[3] = abs(row.iloc[1]-row.iloc[2])  # diff max min
    return row


def fill_null_regression(none_group, out_group, reg_model, col_name, day, K, fill_days=10, index_cols=TRAIN_IND_COLS):
    relevant_indices = none_group.index.unique()
    x = np.random.rand(len(relevant_indices), 4)
    # df we will fill for people with no regression data
    df_cols = [f"{col_name}_{day}_days_median", f"{col_name}_{day}_days_max", f"{col_name}_{day}_days_min", f"{col_name}_{day}_days_min_max_diff"]
    df_to_concat_reg = pd.DataFrame(np.full_like(x, np.nan, dtype=np.double), columns=df_cols, index=relevant_indices)

    # takes only num of days for regression
    relevant_out_group = out_group[out_group[TIME_COL] <= fill_days * 24].set_index(index_cols)

    # indices that are both in the non group and we have data on in X num of days
    sorted_out_group_index = relevant_out_group.sort_index().index
    relevant_indices = [ind for ind in relevant_indices if ind in sorted_out_group_index]
    relevant_out_group = relevant_out_group.loc[relevant_indices, :]

    ind_to_add = [ind for ind in relevant_indices if ind not in reg_model.index]
    reg_model = reg_model.append(pd.Series([np.nan] * len(ind_to_add), index=ind_to_add, name='regression_model'))
    reg_rel = reg_model.loc[relevant_indices].reset_index()
    if len(reg_rel) > 0:
        reg_rel['regression_model'] = reg_rel.apply(lambda row: row['regression_model'] if pd.notnull(row['regression_model']) else calc_regression(pd.Series(relevant_out_group.loc[tuple(row[index_cols]), TIME_COL]), pd.Series(relevant_out_group.loc[tuple(row[index_cols]), "valuenum"]), K), axis=1)
    reg_rel = reg_rel.set_index(index_cols)
    df_to_concat_reg = df_to_concat_reg.apply(lambda x: fill_null_by_reg_for_patient(x, reg_rel), axis=1)
    return df_to_concat_reg


def add_12_hours_lab_result(patient_info, reg_cols_small, reg_cols_big):
    if reg_cols_small is not None and patient_info in reg_cols_small.index and pd.notnull(reg_cols_small[patient_info]): # has first_day days regression
        return reg_cols_small[patient_info].predict([[(TIME_LIMIT - 12)]])[0]
    if reg_cols_big is not None and patient_info in reg_cols_big.index and pd.notnull(reg_cols_big[patient_info]): # has first_day + 2 days regression
        return reg_cols_big[patient_info].predict([[(TIME_LIMIT - 12)]])[0]
    return np.nan


def create_12h_features(feature_group, col_name, reg_cols_dict, first_day, index_cols):
    got_half_day = feature_group[(feature_group[TIME_COL] <= -(TIME_LIMIT - 13)) & (feature_group[TIME_COL] >= -(TIME_LIMIT - 11))]
    add_half_day = feature_group[~feature_group["identifier"].isin(got_half_day["identifier"])].set_index(index_cols)
    got_half_day = get_last_value_df(got_half_day, f"{col_name}_12_hours", index_cols)
    got_half_day, _ = get_is_imputed_cols(got_half_day, got_half_day.columns)
    map_half_day = map(lambda ind: add_12_hours_lab_result(ind, reg_cols_dict[first_day], reg_cols_dict[first_day+2]), add_half_day.index.unique())
    add_half_day = pd.DataFrame(map_half_day, index=add_half_day.index.unique(), columns=[f"{col_name}_12_hours"])
    return pd.concat([got_half_day, add_half_day])


def create_timeframes_comparison_features(feature_df, col_name, first_day):
    col_names = feature_df.columns
    # adding coef ratio between 3 and 5 days time frames if the data exists
    if f"{col_name}_{first_day + 2}_days_reg_coef" in col_names and f"{col_name}_{first_day}_days_reg_coef" in col_names:
        feature_df[f"{col_name}_coef_ratio"] = feature_df[f"{col_name}_{first_day + 2}_days_reg_coef"] / feature_df[f"{col_name}_{first_day}_days_reg_coef"]
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # adding diff in median between 3 and 5 days time frames if the data exists
    if f"{col_name}_{first_day + 2}_days_median" in col_names and f"{col_name}_{first_day}_days_median" in col_names:
        feature_df[f"{col_name}_median_diff"] = feature_df[f"{col_name}_{first_day + 2}_days_median"] - feature_df[
            f"{col_name}_{first_day}_days_median"]


def create_regression_models_and_impute(patient_gb, feature_group, timeframe_group, reg_cols_dict, col_name, day, first_day, index_cols, n=5):
    if day == "all":  # all_days, we don't use regression
        return pd.DataFrame([]), pd.DataFrame([], columns=[f"{col_name}_{day}_days_reg_coef", f"{col_name}_{day}_days_reg_r2", "regression_model"])

    reg_cols = patient_gb.apply(func=lambda x: pd.Series(get_regression(x.hours_from_charttime_time_to_targettime, x.valuenum, n), index=[f"{col_name}_{day}_days_reg_coef", f"{col_name}_{day}_days_reg_r2", "regression_model"]))
    # Taking all patients that has the feature values, but not in the relevant timeframe
    none_group = feature_group[~feature_group["identifier"].isin(timeframe_group["identifier"])].set_index(index_cols)
    if day == first_day + 2:  # filling missing values by linear regression
        reg_cols_dict[first_day + 2] = reg_cols["regression_model"]
        reg_series = pd.Series([None] * len(none_group.index.unique()), name="regression_model", index=none_group.index.unique(), dtype=float)
        none_group_values = fill_null_regression(none_group, feature_group, reg_series, col_name, day, n, index_cols=index_cols)
    else: # day == first_day:
        reg_cols_dict[first_day] = reg_cols["regression_model"]
        none_group_values = fill_null_regression(none_group, feature_group, reg_cols_dict[first_day + 2], col_name, day, n, index_cols=index_cols)

    return none_group_values, reg_cols


def create_time_series(patient_gb, col_name, day):
    df_to_concat = patient_gb.agg({"valuenum": [(f"{col_name}_{day}_days_median", "median"),
                                                (f"{col_name}_{day}_days_std", "std"),
                                                (f"{col_name}_{day}_days_max", "max"),
                                                (f"{col_name}_{day}_days_min", "min"),
                                                (f"{col_name}_{day}_days_min_max_diff", lambda x: x.max() - x.min())]})
    df_to_concat.columns = df_to_concat.columns.get_level_values(1)
    df_to_concat, _ = get_is_imputed_cols(df_to_concat, df_to_concat.columns)  # adding mask features
    return df_to_concat


def not_included(col):
    if any([word in col for word in ['_days_', 'time_from_last', 'ratio', 'diff', '12_hours', "_is_imputed", "_over_time", "fraction_fever_measurements"]]):
        return False
    return True


def create_lab_time_series_features(df, agg_col, stat_dict=None, index_cols=TRAIN_IND_COLS, first_day=3,n=5):
    to_concat, cols_for_existence, new_stat_dict = [], [], {}
    gb = df.groupby(agg_col)

    for col_name, feature_group in gb:
        do_regression = True
        reg_cols_dict = {first_day + 2: None, first_day: None}
        new_stat_dict.update(remove_extreme_values(feature_group, 'valuenum', col_name, stat_dict))
        feature_group.dropna(subset=['valuenum'], inplace=True)  # removing rows with null values
        if len(feature_group) == 0:  # if we got an empty dataframe after removing null
            continue

        last_val = get_last_value_df(feature_group, col_name, index_cols)
        to_concat.append(last_val)
        cols_for_existence.append(col_name)

        feature_to_concat = []
        for day in ["all", first_day+2, first_day]:  # creating time series features for 3 and 5 days
            timeframe_group = feature_group if day == 'all' else feature_group[feature_group[TIME_COL] <= day * 24]
            if len(timeframe_group) == 0:  # if the patient is missing data for one of the time frames the flag = False
                continue
            patient_gb = timeframe_group.groupby(index_cols)
            df_to_concat = create_time_series(patient_gb, col_name, day)
            if day == 'all':
                feature_counts = patient_gb["valuenum"].count().reset_index()
                do_regression = np.quantile(feature_counts.set_index(index_cols)['valuenum'], 0.25, axis=0) >= n

            if do_regression:
                none_group_values, reg_cols = create_regression_models_and_impute(patient_gb, feature_group, timeframe_group, reg_cols_dict, col_name, day, first_day, index_cols, n)
                df_to_concat = pd.concat([df_to_concat, none_group_values])
                feature_to_concat += [df_to_concat, reg_cols.iloc[:, :-1]]
            else:
                feature_to_concat.append(df_to_concat)

        feature_to_concat = [df for df in feature_to_concat if len(df) > 0]
        feature_df = feature_to_concat[0].join(feature_to_concat[1:], how='outer').dropna(how='all', axis=1)
        create_timeframes_comparison_features(feature_df, col_name, first_day)

        half_day_df = create_12h_features(feature_group, col_name, reg_cols_dict, first_day, index_cols)
        feature_df = pd.concat([feature_df, half_day_df], axis=1)

        # removing extreme values after imputation by regression
        org_cols = [col for col in feature_df.columns if re.search("([5432]_days_(min|max|median)$)|(_12_hours$)", col) is not None]
        for col in org_cols:
            remove_extreme_values(feature_df, col, col_name, new_stat_dict)
        to_concat.append(feature_df)

        # need to do by patient
        count_df = feature_group.groupby(index_cols)['valuenum'].count().to_frame(name=f"count_{col_name}_over_time")
        to_concat.append(count_df)

    to_concat = [df for df in to_concat if len(df) > 0]
    all_lab_df = to_concat[0].join(to_concat[1:], how='outer').reset_index().dropna(how='all', axis=1)
    cols_for_existence = [col for col in cols_for_existence if col in all_lab_df.columns and not_included(col)]
    return all_lab_df, cols_for_existence, new_stat_dict


def get_final_lab_df(lab_df, day_to_check=3):
    all_lab_df, cols_for_existence, stat_dict = create_lab_time_series_features(lab_df, "label", first_day=day_to_check)
    lab_cont_attributes = [col for col in all_lab_df.columns if col not in TRAIN_IND_COLS and "_is_imputed" not in col]
    return all_lab_df, lab_cont_attributes, cols_for_existence, stat_dict