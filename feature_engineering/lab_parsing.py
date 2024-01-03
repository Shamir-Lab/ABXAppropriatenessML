from utilities import TIME_LIMIT, TIME_COL, unite_features, IND_COLS

URINE_ITEMIDS = [51094, 51464, 51478, 51484, 51491, 51492, 51493, 51498, 51516]
ASCITES_ITEMIDS = [51116, 51120, 51125, 51127, 51128]


def parse_lab_features(df):
    # parsing Temp in F to temp in C
    temp_f_indices = df[df['label'].isin(["Temperature F", "Temperature F (calc)", "Temperature Fahrenheit"])].index
    df.loc[temp_f_indices, 'valuenum'] = df.loc[temp_f_indices, 'valuenum'].apply(lambda x: (x - 32) * (5 / 9))

    df['label'] = df.apply(lambda r: f'{r["label"]} ({r["itemid"]})', axis=1)
    df.loc[temp_f_indices, 'label'] = "Temperature C"
    unite_features(df, 'label')
    df['got_urine_test'] = df['itemid'].apply(lambda x: int(x in URINE_ITEMIDS))
    df['got_ascites_test'] = df['itemid'].apply(lambda x: int(x in ASCITES_ITEMIDS))
    df.drop(columns=['itemid'], inplace=True)


def get_fever_features(raw_lab_df, agg_col, index_cols=IND_COLS):
    df = raw_lab_df[raw_lab_df[agg_col] == 'Temperature C']
    fever_df = df[df["valuenum"] > 37.5]
    fever_stats = df.groupby(index_cols)['valuenum'].count().rename("count_temp_measurements").to_frame()
    fever_stats["fever_count"] = fever_df.groupby(index_cols)[TIME_COL].count()
    fever_stats['fever_count'].fillna(0, inplace=True)
    fever_stats[f"fraction_fever_measurements"] = fever_stats[f"fever_count"]/fever_stats[f"count_temp_measurements"]

    fever_stats = fever_stats.drop(columns=['fever_count', 'count_temp_measurements']).reset_index()
    fever_attributes = ["fraction_fever_measurements"]
    return fever_stats, fever_attributes


def get_lab_df(lab_df, index_cols=IND_COLS):
    lab_df = lab_df[lab_df[TIME_COL] >= -TIME_LIMIT]
    parse_lab_features(lab_df)
    lab_df = lab_df.drop_duplicates()
    fever_df, fever_attributes = get_fever_features(lab_df, "label", index_cols)
    return lab_df, fever_df, fever_attributes