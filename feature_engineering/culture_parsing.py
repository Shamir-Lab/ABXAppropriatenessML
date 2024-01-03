import numpy as np
import json
import pandas as pd
import os
from utilities import IND_COLS, TIME_LIMIT, TIME_COL, create_existance_features_per_col

culture_site_json = "mimic_culture_site_dict.json"
with open(os.path.join("utilities_files", culture_site_json), "r") as f_json:
    mimic_culture_site_dict = json.load(f_json)

culture_org_json = "mimic_culture_org_dict.json"
with open(os.path.join("utilities_files", culture_org_json), "r") as f_json:
    mimic_culture_org_dict = json.load(f_json)

culture_gram_json = "mimic_culture_gram_dict.json"
with open(os.path.join("utilities_files", culture_gram_json), "r") as f_json:
    gram_dict = json.load(f_json)


def parse_culture_features(df):
    df['spec_type_desc'] = df['spec_type_desc'].map(mimic_culture_site_dict)
    df['org_name'] = df['org_name'].map(mimic_culture_org_dict)
    df['parsed_culture'] = df.apply(lambda r: f"{r['org_name']} from {r['spec_type_desc']}" if pd.notnull(r['spec_type_desc']) and pd.notnull(r['org_name']) else np.nan, axis=1)
    df.dropna(subset=['spec_type_desc', 'org_name', "parsed_culture"], inplace=True)
    df['spec_type_desc'] = df['spec_type_desc'].apply(lambda x: f"Culture from {x}")
    df['Gram'] = df['org_name'].map(gram_dict)
    df.drop_duplicates(inplace=True)


def get_culture_df(c_df, index_cols=IND_COLS):
    # filtering out all entries of cultures that were taken less then 3 days prior prediction time
    c_df = c_df[c_df[TIME_COL] >= -(TIME_LIMIT - 24 * 3)].drop(columns=[TIME_COL])
    parse_culture_features(c_df)
    c_df, culture_attributes = create_existance_features_per_col(c_df, ['spec_type_desc', 'org_name', 'parsed_culture', 'Gram'], index_cols)
    return c_df, culture_attributes[:-1], [culture_attributes[-1]]  # last attribute is continuous


def get_culture_antibiotic_df(ca_df, index_cols=IND_COLS):
    # filtering out all entries of cultures that were taken less then 3 days prior prediction time
    ca_df = ca_df[ca_df[TIME_COL] >= -(TIME_LIMIT - 24 * 3)].drop(columns=[TIME_COL])
    ca_df['org_name'] = ca_df['org_name'].map(mimic_culture_org_dict)
    sub_df = ca_df[ca_df['interpretation'] == 'R']
    sub_df = sub_df.groupby(index_cols)['interpretation'].count().rename("Total_Resistant_Culture")

    ca_df['parsed_antibiotic'] = ca_df.apply(lambda r: f"{r['ab_name']} - {r['interpretation']}" if pd.notnull(r['ab_name']) and pd.notnull(r['interpretation']) else np.nan, axis=1)
    ca_df['parsed_culture_antibiotic'] = ca_df.apply(lambda r: f"{r['org_name']} - {r['ab_name']} - {r['interpretation']}" if pd.notnull(r['org_name']) and pd.notnull(r['ab_name']) and pd.notnull(r['interpretation']) else np.nan, axis=1)
    ca_df.drop(columns=['org_name', 'ab_name', 'interpretation'], inplace=True)
    ca_df, culture_antibiotic_attributes = create_existance_features_per_col(ca_df, ['parsed_culture_antibiotic', 'parsed_antibiotic'], index_cols)

    ca_df = ca_df.set_index(index_cols).join(sub_df, how='outer').reset_index()
    return ca_df, culture_antibiotic_attributes, ["Total_Resistant_Culture"]
