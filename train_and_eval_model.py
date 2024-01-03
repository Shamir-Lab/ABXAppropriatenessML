from utilities import IND_COLS, TRAIN_IND_COLS
from feature_engineering.processing_dataset import final_parsing_train, final_parsing_validation
from feature_engineering.feature_parsing import parse_data_train, parse_data_validation, FILE_NAMES
import dill as pickle
import json
import pandas as pd
import os
from models.ABX_model import ABXModel
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


def train_final_model(args, run_parsing=False, day_to_check=2, file_names=FILE_NAMES):
    if run_parsing:
        out_dir, pkl_name = os.path.split(args.pkl_path)
        parse_data_train(args.input_dir, out_dir, file_names=file_names, pkl_name=pkl_name)

    with open(args.pkl_path, "rb") as f_in:
        data_objs = [pickle.load(f_in) for ob in range(5)]
        merged, raw_lab_df, cat_attributes, cont_attributes, cont_cols_for_fillna = data_objs

    df, n_cat_attributes, n_cont_attributes, cont_cols_for_fillna, cols_for_existence, stat_dict = final_parsing_train(merged, raw_lab_df, cat_attributes, cont_attributes, cont_cols_for_fillna, index_cols=TRAIN_IND_COLS, day_to_check=day_to_check)
    to_save = [cols_for_existence, cont_cols_for_fillna, n_cat_attributes, n_cont_attributes, stat_dict, day_to_check]
    X_train, y_train = df.drop('target', axis=1), df['target']
    with open(args.param_json, 'rb') as fin:
        param_dict = json.load(fin)
    model = ABXModel(args.model_name, args.features_choice, args.K, param_dict, args.imp, args.norm, n_cont_attributes, n_cat_attributes, data_ensemble=args.data_ensemble, null_thresh=args.null_thresh, corr_threshold=args.corr_threshold, n_keep=args.n_keep, balancing=args.balancing, balancing_ratio=args.balancing_ratio)
    model.fit(X_train, y_train)

    with open(args.output_pkl, 'wb') as fout:
        [pickle.dump(ob, fout) for ob in to_save]
        pickle.dump(model, fout)


def run_final_model(input_dir, data_pkl, model_pkl, output_file, label_file, label_col, run_parsing=False, file_names=FILE_NAMES):
    if run_parsing:
        out_dir, pkl_name = os.path.split(data_pkl)
        parse_data_validation(input_dir, out_dir, file_names=file_names, pkl_name=pkl_name)

    with open(data_pkl, "rb") as f_in:
        data_objs = [pickle.load(f_in) for ob in range(2)]
        merged, raw_lab_df = data_objs

    with open(model_pkl, 'rb') as f_in:
        data_objs = [pickle.load(f_in) for ob in range(7)]
        train_cols_for_existence, train_cont_cols_for_fillna, train_cat_attributes, train_cont_attributes, stat_dict, day_to_check, model = data_objs

    X_test = final_parsing_validation(merged, raw_lab_df, train_cols_for_existence, train_cont_cols_for_fillna, train_cont_attributes, train_cat_attributes, stat_dict, day_to_check=day_to_check)
    X_test.set_index(list(IND_COLS), inplace=True)
    y_test = pd.read_csv(label_file).set_index(list(IND_COLS))
    X_test = X_test.loc[y_test.index] # In case some are missing label
    y_test = y_test.loc[:, label_col]
    res_df = model.evaluation(X_test, y_test)

    precision, recall, thresh = precision_recall_curve(res_df['target'], res_df['score'])
    prauc_score = auc(recall, precision)
    rocauc_score = roc_auc_score(res_df['target'], res_df['score'])
    print(rocauc_score)
    print(prauc_score)

    res_df.to_csv(output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Training final model and saving it to pickle file')
    parser.add_argument('--pkl_path', help='Path to pickle file with raw data objects')
    parser.add_argument('--param_json', help='Path to json file with model hyperparameters')
    parser.add_argument('--output_pkl', help='Path to pickle file to save the model and the relevant objects')
    parser.add_argument('--model_name', help='name of model')
    parser.add_argument('--features_choice', help='name of features_choice')
    parser.add_argument('--balancing', default='BorderlineSMOTE', help='name of balancing method, default="BorderlineSMOTE"')
    parser.add_argument('--balancing_ratio', type=float, help='ratio for balancing. default: None', default=None)
    parser.add_argument('--imp', help='name of imputation method')
    parser.add_argument('--norm', help='name of normalization method')
    parser.add_argument('--K', '-K', type=int, help='number of features to select in shap and selectKBest')
    parser.add_argument('--null_thresh', type=float, default=0.7, help='Null threshold for features. default:0.7')
    parser.add_argument('--corr_threshold', type=float, default=0.7, help='correlation filtration threshold for features. default:0.7')
    parser.add_argument('--n_keep', type=int, default=1, help='Number of features to keep from the same raw measurement. default=1')
    parser.add_argument('--data_ensemble', action='store_true', help='whether to use BalancedDataEnsemble. Default is False')
    parser.set_defaults(data_ensemble=False)
    args = parser.parse_args()

    train_final_model(args, run_parsing=False, day_to_check=2)

