import pandas as pd
from patient_parsing import get_patient_df
from drugs_parsing import get_drug_df
from culture_parsing import get_culture_df, get_culture_antibiotic_df
from procedures_parsing import get_procedure_df
from lab_parsing import get_lab_df
from utilities import TIME_COL, IND_COLS, TRAIN_IND_COLS
from functools import reduce
import os
import pickle


FILE_NAMES = ('patient.csv', 'drug.csv', 'culture.csv', 'antibiotic.csv', 'procedure.csv', 'lab.csv')


def get_mimic_pre_processing(p_df, drg_df, c_df, ca_df, pr_df, lab_df, index_cols=IND_COLS, is_validation=False):
    """
    :param p_df: df with raw patient information.
    :param drg_df: df with raw drugs information.
    :param c_df: df with raw culture information.
    :param ca_df: df with information about culture paired with antibiotic checked on it.
    :param pr_df: df with raw procedure information.
    :param lab_df: df with raw lab and vital signs information.
    :param index_cols: list of columns to use as index.
    :param is_validation: whether we are parsing vaidation set or training set.
    """
    p_df, patient_cont_features, patient_cat_features = get_patient_df(p_df)
    all_drugs_df, drug_cont_attributes = get_drug_df(drg_df, index_cols)
    c_df, culture_cat_features, cont_culture_features = get_culture_df(c_df, index_cols)
    ca_df, culture_antibiotic_cat_features, cont_antibiotic_features = get_culture_antibiotic_df(ca_df, index_cols)
    pr_df, procedures_attributes = get_procedure_df(pr_df, index_cols)
    raw_lab_df, fever_df, fever_attributes = get_lab_df(lab_df, index_cols)
    # creating a binary feature of whether the patient had any ascites or urine lab tests.
    lab_test_df = raw_lab_df.groupby(index_cols)[['got_urine_test', 'got_ascites_test']].max().reset_index()
    raw_lab_df = raw_lab_df[raw_lab_df['label'] != "Ascites"].sort_values(['identifier', 'label', 'valuenum', TIME_COL])

    data_frames = [all_drugs_df, c_df, ca_df, p_df, pr_df, lab_test_df, fever_df]
    merged = reduce(lambda left, right: pd.merge(left, right, on=index_cols, how='outer'), data_frames)

    if is_validation:
        # filtering patient and culture features that had less than 4% positive values
        filtered_culture_procedures = [col for col in culture_cat_features + culture_antibiotic_cat_features + procedures_attributes if merged[col].sum() / len(merged) >= 0.04]
        cat_attributes = patient_cat_features + filtered_culture_procedures + ['got_urine_test', 'got_ascites_test']
    else:
        cat_attributes = patient_cat_features + culture_cat_features + culture_antibiotic_cat_features + procedures_attributes + ['got_urine_test', 'got_ascites_test']
    cont_attributes = patient_cont_features + drug_cont_attributes + cont_culture_features + cont_antibiotic_features + fever_attributes
    cont_cols_for_fillna = drug_cont_attributes + cont_culture_features + cont_antibiotic_features + fever_attributes

    return raw_lab_df.drop(columns=(['got_urine_test', 'got_ascites_test'])), merged.sort_values('identifier'), sorted(cat_attributes), sorted(cont_attributes), sorted(cont_cols_for_fillna)


def get_input_dfs(input_dir, file_names=FILE_NAMES):
    return [pd.read_csv(os.path.join(input_dir, file_name)) for file_name in file_names]


def parse_data_train(input_dir, out_dir, file_names=FILE_NAMES, pkl_name="train_raw_data.pkl"):
    input_dfs = get_input_dfs(input_dir, file_names)
    raw_lab_df, merged, cat_attributes, cont_attributes, cont_cols_for_fillna = get_mimic_pre_processing(*input_dfs, index_cols=TRAIN_IND_COLS)

    merged = merged.set_index(TRAIN_IND_COLS).round(6).reset_index()
    raw_lab_df = raw_lab_df.set_index(TRAIN_IND_COLS).round(6).reset_index()
    to_save = [merged, raw_lab_df, cat_attributes, cont_attributes, cont_cols_for_fillna]
    with open(os.path.join(out_dir, pkl_name), "wb") as f_out:
        for obj in to_save:
            pickle.dump(obj, f_out)


def parse_data_validation(input_dir, out_dir, file_names=FILE_NAMES, pkl_name="validation_raw_data.pkl"):
    input_dfs = get_input_dfs(input_dir, file_names)
    raw_lab_df, merged, _, _, _ = get_mimic_pre_processing(*input_dfs, index_cols=IND_COLS, is_validation=True)

    merged = merged.set_index(IND_COLS).round(6).reset_index()
    raw_lab_df = raw_lab_df.set_index(IND_COLS).round(6).reset_index()
    to_save = [merged, raw_lab_df]
    with open(os.path.join(out_dir, pkl_name), "wb") as f_out:
        for obj in to_save:
            pickle.dump(obj, f_out)