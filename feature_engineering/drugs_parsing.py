import json
import os
from utilities import IND_COLS, TIME_LIMIT, TIME_COL


drugs_json = "mimic_drug_to_label.json"
with open(os.path.join("utilities_files", drugs_json), "r") as f_json:
    drugs_dict = json.load(f_json)


def process_drug_df(df, agg_col, index_cols=IND_COLS):
    to_concat = []

    gb = df.groupby('drug_cat')
    for col_name, out_group in gb:
        for day in ["all", 2, 3, 4, 5]:
            if day == "all":
                relevant_group = out_group
                count_col = f"{col_name}_count"
            else:
                relevant_group = out_group[out_group[TIME_COL] <= day * 24]
                count_col = f"{col_name}_{day}_days_count"

            ggb = relevant_group.groupby(index_cols)
            df_to_concat = ggb.agg({agg_col: [(count_col, "count")]})
            df_to_concat.columns = df_to_concat.columns.get_level_values(1)

            if not df_to_concat.empty:
                to_concat.append(df_to_concat)

    all_drugs_df = to_concat[0].join(to_concat[1:], how='outer').reset_index()
    drug_cont_attributes = [col for col in all_drugs_df.columns if col not in index_cols + ['drug_cat']]
    return all_drugs_df, drug_cont_attributes


def get_drug_df(drg_df, index_cols=IND_COLS):
    # filtering out entries of drugs that were administered after our prediction time
    drg_df = drg_df[drg_df[TIME_COL] >= -TIME_LIMIT]
    drg_df['drug_cat'] = drg_df["itemid"].apply(lambda x: drugs_dict[str(x)])
    all_drugs_df, drug_cont_attributes = process_drug_df(drg_df, "itemid", index_cols)
    return all_drugs_df, drug_cont_attributes
