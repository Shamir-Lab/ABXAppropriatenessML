import numpy as np
import json
import pandas as pd
import os


ethnicity_json = "ethnicity_dict.json"
with open(os.path.join("utilities_files", ethnicity_json)) as f_json:
    ethnicity_dict = json.load(f_json)

ETHNICITIES = list(set(ethnicity_dict.values()))


def BMI(data):
    if not isinstance(data['admission_weight'], float) or not isinstance(data['admission_height'], float) or data['admission_height'] == 0:
        return np.nan
    return data['admission_weight'] / (data['admission_height'] / 100)**2


def parse_patient_features(df):
    df['gender'] = df['gender'].map({"M": 1, "F": 0})

    # parsing inches to cm
    df['admission_height'] = df['admission_height'].apply(lambda x: x * 2.54 if pd.notnull(x) else x)

    # get rid of invalid values
    df['age'] = df['age'].apply(lambda x: 99 if x == 300 else x)
    df['age'] = df['age'].apply(lambda x: np.nan if x > 120 else x)
    df['admission_height'] = df['admission_height'].apply(lambda x: np.nan if pd.isnull(x) or x > 250 or x < 20 else x)
    df['BMI'] = df.apply(BMI, axis=1)

    df['ethnicity'] = df['ethnicity'].map(ethnicity_dict)
    for ethnicity in ETHNICITIES:
        df[f"is_{ethnicity}"] = df['ethnicity'].apply(lambda x: 1 if x == ethnicity else 0)


def get_patient_df(p_df):
    parse_patient_features(p_df)
    patient_attributes = ['age', 'admission_weight', 'admission_height', 'BMI', 'hours_from_admittime_to_targettime', 'hours_from_icutime_to_targettime']
    ethnicity_cats = [f"is_{ethnicity}" for ethnicity in ETHNICITIES]
    patient_categorical_attributes = ['gender'] + ethnicity_cats
    return p_df, patient_attributes, patient_categorical_attributes
