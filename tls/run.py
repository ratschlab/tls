# -*- coding: utf-8 -*-
import argparse
import functools
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd

from tls.common.constants import MORTALITY_NAME, CIRC_FAILURE_NAME, RESP_FAILURE_NAME, URINE_REG_NAME, \
    URINE_BINARY_NAME, PHENOTYPING_NAME, LOS_NAME
from tls.common.datasets import Dataset
from tls.common.lookups import read_var_ref_table
from tls.common.processing import map_df
from tls.common.reference_data import read_static
from tls.common.resampling import irregular_to_gridded
from tls.data import imputation_for_endpoints, extended_general_table_generation, endpoint_generation, \
    labels, schemata
from tls.data.feature_extraction import extract_feature_df
from tls.data.preprocess import to_ml, run_non_scaled_pipe, scaling_data_common
from tls.models.train import train_with_gin
from tls.models.utils import get_bindings_and_params
from tls.preprocessing import merge

default_seed = 42


def build_parser():
    parser = argparse.ArgumentParser(
        description='Temporal label smoothing for ICU.')

    parent_parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(title='Commands',
                                       dest='command', required=True)

    parser_prep_ml = subparsers.add_parser('preprocess',
                                           help='Calls sequentially merge and resample.',
                                           parents=[parent_parser])

    preprocess_arguments = parser_prep_ml.add_argument_group(
        'Preprocess arguments')

    preprocess_arguments.add_argument('--work-dir',
                                      required=False, type=Path,
                                      help="")
    preprocess_arguments.add_argument('--hirid-data-root',
                                      type=Path,
                                      required=False,
                                      help="Path to the decompressed parquet data directory as published on physionet.")
    preprocess_arguments.add_argument('--resource-path', dest="resource_path",
                                      required=True, type=Path,
                                      help="Path to load the resources from ")
    preprocess_arguments.add_argument('-nw', '--nr-workers', default=1,
                                      required=False, type=int,
                                      dest='nr_workers',
                                      help='Number of process to use at preprocessing, Default to 1 ')
    preprocess_arguments.add_argument('--split-path', dest="split_path",
                                      default=None, required=False, type=Path,
                                      help="Path to load the data split from from ")
    preprocess_arguments.add_argument('--clipping', dest="clipping",
                                      default=0.0, required=False, type=float,
                                      help="Quantile to clip before scaling")
    preprocess_arguments.add_argument('--scaler', dest="scaler",
                                      default='standard', required=False, type=str,
                                      help="Type of scaler to use for continuous columns")
    preprocess_arguments.add_argument('--seed', dest="seed",
                                      default=default_seed, required=False, type=int,
                                      help="Seed for the train/val/test split")
    preprocess_arguments.add_argument('--imputation', dest="imputation",
                                      default='ffill', required=False, type=str,
                                      help="Type of imputation. Default: 'ffill' ")
    preprocess_arguments.add_argument('--horizons', dest="horizons",
                                      default=12, required=False, type=int, nargs='+',
                                      help="Horizons of prediction in hours for failure tasks")
    preprocess_arguments.add_argument("--dataset", default='hirid', required=False, type=str,
                                      help="Name of the dataset to build, only 'mimic3' ")
    preprocess_arguments.add_argument("--mimic3-data-root", type=Path, required=False,
                                      help="Path to folder containing MIMIC-III Benchmark source path")
    preprocess_arguments.add_argument("--mimic3-static-columns", nargs='+', help="List of static columns names ")

    model_arguments = parent_parser.add_argument_group('Model arguments')
    model_arguments.add_argument('-l', '--logdir', dest="logdir",
                                 required=False, type=str,
                                 help="Path to the log directory ")
    model_arguments.add_argument('--reproducible', default=True, dest="reproducible",
                                 required=False, type=str,
                                 help="Whether to configure torch to be reproducible.")
    model_arguments.add_argument('-sd', '--seed', default=1111, dest="seed",
                                 required=False, nargs='+', type=int,
                                 help="Random seed at training and evaluation, default : 1111")
    model_arguments.add_argument('-t', '--task', default=None, dest="task",
                                 required=False, nargs='+', type=str,
                                 help="Name of the task : Default None")
    model_arguments.add_argument('-r', '--resampling', default=None, dest="res",
                                 required=False, type=int,
                                 help="resampling for the data")
    model_arguments.add_argument('-rl', '--resampling_label', default=None,
                                 dest="res_lab", required=False, type=int,
                                 help="resampling for the prediction")
    model_arguments.add_argument('-bs', '--batch-size', default=None,
                                 dest="batch_size", required=False,
                                 type=int, nargs='+',
                                 help="Batchsize for the model")
    model_arguments.add_argument('-lr', '--learning-rate', default=None, nargs='+',
                                 dest="lr", required=False, type=float,
                                 help="Learning rate for the model")
    model_arguments.add_argument('--lr-decay', default=None, nargs='+',
                                 dest="lr_decay", required=False, type=float,
                                 help="Learning rate decay at every epoch for exponential decay schedule")
    model_arguments.add_argument('--maxlen', default=None, dest="maxlen",
                                 required=False, type=int,
                                 help="Max length of considered time-series for the model")
    model_arguments.add_argument('--num-class', default=None, dest="num_class",
                                 required=False, type=int,
                                 help="Number of classes considered for the task")
    model_arguments.add_argument('-emb', '--emb', default=None, dest="emb",
                                 required=False, nargs='+', type=int,
                                 help="Embedding size of the input data")
    model_arguments.add_argument('-kernel', '--kernel', default=None,
                                 dest="kernel", required=False, nargs='+',
                                 type=int, help="Kernel size for Temporal CNN")
    model_arguments.add_argument('-do', '--do', default=None, dest="do",
                                 required=False, nargs='+', type=float,
                                 help="Dropout probability for the Transformer block")
    model_arguments.add_argument('-do_att', '--do_att', default=None, dest="do_att",
                                 required=False, nargs='+', type=float,
                                 help="Dropout probability for the Self-Attention layer only")
    model_arguments.add_argument('-depth', '--depth', default=None,
                                 dest="depth", required=False, nargs='+',
                                 type=int,
                                 help="Number of layers in Neyral Network")
    model_arguments.add_argument('-heads', '--heads', default=None,
                                 dest="heads", required=False, nargs='+',
                                 type=int,
                                 help="Number of heads in Sel-Attention layer")
    model_arguments.add_argument('-latent', '--latent', default=None,
                                 dest="latent", required=False, nargs='+',
                                 type=int,
                                 help="Dimension of fully-connected layer in Transformer block")
    model_arguments.add_argument('-ed', '--embedding-depth', default=None,
                                 dest="embedding_depth", required=False, nargs='+',
                                 type=int,
                                 help="Depth of the embedding module")
    model_arguments.add_argument('-horizon', '--horizon', default=None,
                                 dest="horizon", required=False, nargs='+',
                                 type=int,
                                 help="History length for Neural Networks")
    model_arguments.add_argument('-hidden', '--hidden', default=None,
                                 dest="hidden", required=False, nargs='+',
                                 type=int,
                                 help="Dimensionality of hidden layer in Neural Networks")
    model_arguments.add_argument('--subsample-data', default=None,
                                 dest="subsample_data", required=False, nargs='+',
                                 type=float,
                                 help="Subsample parameter in Gradient Boosting, subsample ratio of the training instance")
    model_arguments.add_argument('--subsample-feat', default=None,
                                 dest="subsample_feat", required=False, nargs='+',
                                 type=float,
                                 help="Colsample_bytree parameter in Gradient Boosting, subsample ratio of columns when constructing each tree")
    model_arguments.add_argument('--reg', default=None,
                                 dest="reg", required=False, nargs='+',
                                 type=str,
                                 help="L1 or L2 regularization type")
    model_arguments.add_argument('--reg-weight', default=None,
                                 dest="reg_weight", required=False, nargs='+',
                                 type=float,
                                 help="Regularization strengh")
    model_arguments.add_argument('-rs', '--random-search', default=False,
                                 dest="rs", required=False, type=bool,
                                 help="Random Search setting")
    model_arguments.add_argument('--gamma', default=None, type=str,
                                 dest="gamma", required=False, nargs='+',
                                 help="Power for FocalLoss. Higher means lower weight for confident samples. O is CE")
    model_arguments.add_argument('--smooth-type', default=None, type=str,
                                 dest="smooth_type", required=False, nargs='+',
                                 help="Type of smoothing 'q_step', 'q_linear', 'q_exp' ")
    model_arguments.add_argument('--h-min', default=None, type=int,
                                 dest="h_min", required=False, nargs='+', action='append',
                                 help="min history in number of steps")
    model_arguments.add_argument('--l-smooth', default=None, type=float,
                                 dest="l_smooth", required=False, nargs='+',
                                 help="Strength of q_ls smoothing alpha. Has to be below 0.5")
    model_arguments.add_argument('--up', default=None, type=float,
                                 dest="up", required=False, nargs='+',
                                 help="up weight first step survival TCSR")
    model_arguments.add_argument('--h-max', default=None, type=int, action='append',
                                 dest="h_max", required=False, nargs='+',
                                 help="max history in number of steps")
    model_arguments.add_argument('--h-true', default=None, type=int, action='append',
                                 dest="h_true", required=False, nargs='+',
                                 help="true history in number of steps")
    model_arguments.add_argument('--repeat-step', default=None, type=int,
                                 dest="repeat_step", required=False, nargs='+',
                                 help="Number of time to repeat each step for TCSR ")
    model_arguments.add_argument('--min-value', default=None, type=float,
                                 dest="min_value", required=False, nargs='+',
                                 help="Min value for smoothing in survival likelihood ")
    model_arguments.add_argument('--ad', default=None, type=float,
                                 dest="ad", required=False, nargs='+',
                                 help="Alpha value for DDRSA")
    model_arguments.add_argument('-c_parameter', '--c_parameter', default=None,
                                 dest="c_parameter", required=False, nargs='+',
                                 help="C parameter in Logistic Regression")
    model_arguments.add_argument('-penalty', '--penalty', default=None,
                                 dest="penalty", required=False, nargs='+',
                                 help="Penalty parameter for Logistic Regression")
    model_arguments.add_argument('--loss-weight', default=None,
                                 dest="loss_weight", required=False, nargs='+', type=str,
                                 help="Loss weigthing parameter")
    model_arguments.add_argument('--objective-type', default=None,
                                 dest="objective_type", required=False, nargs='+', type=str,
                                 help="Type of objective")
    model_arguments.add_argument('--agg-type', default=None,
                                 dest="agg_type", required=False, nargs='+', type=str,
                                 help="Loss weigthing aggregation method between class, patient, and event")
    model_arguments.add_argument('-o', '--overwrite', default=False,
                                 dest="overwrite", required=False, type=bool,
                                 help="Boolean to overwrite previous model in logdir")
    model_arguments.add_argument('-c', '--config', default=None, dest="config",
                                 nargs='+', type=str,
                                 help="Path to the gin train config file.")

    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate',
                                            parents=[parent_parser])

    parser_train = subparsers.add_parser('train', help='train',
                                         parents=[parent_parser])
    return parser


def run_merge_step(hirid_path, var_ref_path, merged_path, nr_workers, static_data_path=None, part_nr=None):
    static_data_path = Path(static_data_path) if static_data_path else hirid_path / 'general_table'

    if not Dataset(merged_path).is_done():
        logging.info("Running merge step...")
        merge.merge_tables(
            hirid_path / 'observation_tables' if not part_nr else hirid_path / 'observation_tables' / f'part-{part_nr}.parquet',
            hirid_path / 'pharma_records' if not part_nr else hirid_path / 'pharma_records' / f'part-{part_nr}.parquet',
            static_data_path,
            var_ref_path,
            merged_path,
            nr_workers
        )
    else:
        logging.info(f"Skipping merging, as outputdata seems to exist in {merged_path}")


def run_resample_step(merged_path: Path, static_path, var_ref_path, common_path, nr_workers: int):
    merged_ds = Dataset(merged_path)
    output_ds = Dataset(common_path)

    if output_ds.is_done():
        logging.info(f"Skipping resampling, as outputdata seems to exist in {common_path}")
        return

    logging.info("Running resample step...")
    parts = merged_ds.list_parts()

    df_static = read_static(static_path)
    df_var_ref = read_var_ref_table(var_ref_path)

    prepare_data_fn = functools.partial(irregular_to_gridded,
                                        df_static=df_static,
                                        df_var_ref=df_var_ref)

    output_ds.prepare()

    map_df(prepare_data_fn, parts,
           lambda p: pd.read_parquet(p),
           lambda df, p: df.to_parquet(common_path / p,
                                       index=False), nr_workers)

    output_ds.mark_done()


def run_feature_extraction_step(common_path: Path, var_ref_path, feature_path, nr_workers: int):
    common_ds = Dataset(common_path)
    feature_ds = Dataset(feature_path)

    if feature_ds.is_done():
        logging.info(f"Skipping feature extraction, as outputdata seems to exist in {feature_path}")
        return

    logging.info("Running feature extraction step")
    parts = common_ds.list_parts()

    df_var_ref = read_var_ref_table(var_ref_path)

    prepare_data_fn = functools.partial(extract_feature_df,
                                        df_var_ref=df_var_ref)

    feature_ds.prepare()

    map_df(prepare_data_fn, parts,
           lambda p: pd.read_parquet(p),
           lambda df, p: df.to_parquet(feature_path / p,
                                       index=False), nr_workers)

    feature_ds.mark_done()


def run_build_ml(common_path, labels_path, features_path: Optional[Path], ml_path, var_ref_path,
                 endpoint_names: Sequence[str],
                 imputation: str, seed: int, split_path=None, scaler='standard', clipping=0.0):
    common_ds = Dataset(common_path)
    parts = common_ds.list_parts()

    labels_ds = Dataset(labels_path, part_re=re.compile('batch_([0-9]+).parquet'))
    labels = labels_ds.list_parts()

    if features_path:
        features_ds = Dataset(features_path)
        features = features_ds.list_parts()
    else:
        features = []

    df_var_ref = read_var_ref_table(var_ref_path)

    output_ds = Dataset(ml_path)

    output_cols = schemata.cols_ml_stage_v1

    if not output_ds.is_done():
        logging.info("Running build_ml")
        output_ds.prepare(single_part=True)
        to_ml(ml_path, parts, labels, features, endpoint_names, df_var_ref,
              imputation, output_cols, split_path=split_path,
              random_seed=seed, scaler=scaler, clipping_quantile=clipping)
    else:
        logging.info(f"Data in {ml_path} seem to exist, skipping")


def _get_general_data_path(general_data_path, hirid_data_root):
    if general_data_path:
        return Path(general_data_path)
    else:
        general_data_path_physionet_download = (hirid_data_root / 'general_table.csv')
        if general_data_path_physionet_download.exists():
            return general_data_path_physionet_download

    return hirid_data_root / 'general_table'


def run_hirid_preprocessing_pipeline(hirid_data_root, work_dir, var_ref_path, imputation_method,
                                     general_data_path=None, split_path=None, seed=default_seed, nr_workers=1,
                                     scaler='standard', clipping=0, horizon=12):
    work_dir.mkdir(exist_ok=True, parents=True)

    general_data_path = _get_general_data_path(general_data_path, hirid_data_root)

    extended_general_data_path = work_dir / 'general_table_extended.parquet'

    if imputation_method == 'None':
        imputation_method = None

    if not extended_general_data_path.exists():
        logging.info(f"Generating extended general table in {extended_general_data_path}")

        extended_general_table_generation.generate_extended_general_table(hirid_data_root / 'observation_tables',
                                                                          general_data_path,
                                                                          extended_general_data_path)
    else:
        logging.info(f"Using extended general table in {extended_general_data_path}")

    merged_path = work_dir / 'merged_stage'

    imputation_for_endpoints_path = work_dir / 'imputation_for_endpoints'
    endpoints_path = work_dir / 'endpoints'
    common_path = work_dir / 'common_stage'
    if isinstance(horizon, list):
        label_name = "_".join(['labels', '_'.join([str(h) for h in horizon])]) + 'h'
        ml_name = '_'.join(('imputation', str(imputation_method), 'scaler', scaler,
                            'clip', str(clipping), 'ml_stage', '_'.join([str(h) for h in horizon]) + 'h' + '.h5'))
    elif isinstance(horizon, int):
        label_name = "_".join(['labels', str(horizon)]) + 'h'
        ml_name = '_'.join(('imputation', str(imputation_method), 'scaler', scaler,
                            'clip', str(clipping), 'ml_stage', str(horizon) + 'h' + '.h5'))
    else:
        raise Exception('horizon has to be int or iterable')
    label_path = work_dir / label_name

    features_path = work_dir / 'features_stage'

    ml_path = work_dir / 'ml_stage' / ml_name

    run_merge_step(hirid_data_root, var_ref_path, merged_path, nr_workers, extended_general_data_path)

    run_resample_step(merged_path, extended_general_data_path, var_ref_path, common_path, nr_workers)

    if not imputation_for_endpoints_path.exists():
        logging.info("Running imputation step for endpoints")
        imputation_for_endpoints.impute_for_endpoints(merged_path, imputation_for_endpoints_path, nr_workers=nr_workers)
    else:
        logging.info(f"Data for imputation for endpoints in {imputation_for_endpoints_path} seems to exist, skipping")

    if not endpoints_path.exists():
        logging.info("Running endpoint generation")
        endpoint_generation.generate_endpoints(merged_path, imputation_for_endpoints_path, endpoints_path,
                                               nr_workers=nr_workers)
    else:
        logging.info(f"Endpoints in {endpoints_path} seem to exist, skipping")

    if not label_path.exists():
        logging.info("Running label generation")
        labels.generate_labels(endpoints_path, imputation_for_endpoints_path, extended_general_data_path, label_path,
                               nr_workers=nr_workers, horizon=horizon)
    else:
        logging.info(f"Labels in {label_path} seem to exist, skipping")

    run_feature_extraction_step(common_path, var_ref_path, features_path, nr_workers)
    if isinstance(horizon, int):
        horizon_dependant_names = (CIRC_FAILURE_NAME + '_' + str(horizon) + 'Hours',
                                   RESP_FAILURE_NAME + '_' + str(horizon) + 'Hours',
                                   RESP_FAILURE_NAME + '_mid_' + str(horizon) + 'Hours',
                                   RESP_FAILURE_NAME + '_severe_' + str(horizon) + 'Hours')
    elif isinstance(horizon, list) or isinstance(horizon, tuple):
        horizon_dependant_names = []
        for h in horizon:
            horizon_dependant_names += [CIRC_FAILURE_NAME + '_' + str(h) + 'Hours',
                                        RESP_FAILURE_NAME + '_' + str(h) + 'Hours',
                                        RESP_FAILURE_NAME + '_mid_' + str(h) + 'Hours',
                                        RESP_FAILURE_NAME + '_severe_' + str(h) + 'Hours']
        horizon_dependant_names = tuple(horizon_dependant_names)
    else:
        raise Exception('horizon has to be an int or a iterable')

    endpoints = horizon_dependant_names + (MORTALITY_NAME,
                                           CIRC_FAILURE_NAME + '_' + 'Event',
                                           RESP_FAILURE_NAME + '_' + 'Event',
                                           RESP_FAILURE_NAME + '_mid_' + 'Event',
                                           RESP_FAILURE_NAME + '_severe_' + 'Event',
                                           URINE_REG_NAME,
                                           URINE_BINARY_NAME,
                                           PHENOTYPING_NAME,
                                           LOS_NAME)

    run_build_ml(common_path, label_path, features_path, ml_path, var_ref_path, endpoints,
                 imputation_method, seed, split_path, scaler, clipping)


def run_mimic3_preprocessing_pipeline(mimic_data_root, work_dir, channel_to_id,
                                      matching_dict, var_range, static_col, horizon,
                                      scaling_method='standard', clip_p=0.0):
    """Wrapper to build MIMIC-III benchmark dataset in the desired format for our loader.
    Args:
        mimic_data_root: String with path to source data from https://github.com/YerevaNN/mimic3-benchmarks.
        work_dir: String with path where to save the final h5 file.
        channel_to_id: Dict obtained from mimic_resources/ that matches variables to id.
        matching_dict: Dict obtained from mimic_resources/ that matches string to categories.
        var_range: Dict obtained from mimic_resources/ with ranges for certain variables to remove false entries.
        static_col: Name of the static columns, should be only Height.
    Returns:
    """
    unscaled_save_file = Path(os.path.join(work_dir, 'non_scaled.h5'))
    run_non_scaled_pipe(mimic_data_root, unscaled_save_file, channel_to_id, matching_dict, var_range, static_col,
                        horizon)
    if clip_p > 0:
        scaled_save_file = Path(os.path.join(work_dir, '_'.join([scaling_method, 'clip', str(clip_p), 'scaled.h5'])))
    else:
        scaled_save_file = Path(os.path.join(work_dir, '_'.join([scaling_method, 'scaled.h5'])))

    static_idx = [k for k in range(len(static_col))]
    scaling_data_common(unscaled_save_file, scaled_save_file,
                        threshold=25, scaling_method=scaling_method,
                        static_idx=static_idx, df_ref=None,
                        clip_p=clip_p)


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    # Dispatch
    if args.command == 'preprocess':
        if args.dataset == 'hirid':
            var_ref_path = args.resource_path / 'varref.tsv'
            split_path = args.resource_path / 'split.tsv'
            run_hirid_preprocessing_pipeline(args.hirid_data_root, args.work_dir, var_ref_path,
                                             imputation_method=args.imputation,
                                             split_path=split_path,
                                             seed=args.seed, nr_workers=args.nr_workers,
                                             scaler=args.scaler, clipping=args.clipping, horizon=args.horizons)
        elif args.dataset == 'mimic3':
            with open(os.path.join(args.resource_path, 'channel_to_id_m3.pkl'), 'rb') as f:
                channel_to_id = pickle.load(f)
            with open(os.path.join(args.resource_path, 'matching_dict_m3.pkl'), 'rb') as f:
                matching_dict = pickle.load(f)
            with open(os.path.join(args.resource_path, 'var_range_m3.pkl'), 'rb') as f:
                var_range = pickle.load(f)
            run_mimic3_preprocessing_pipeline(args.mimic3_data_root, args.work_dir, channel_to_id=channel_to_id,
                                              matching_dict=matching_dict, var_range=var_range,
                                              static_col=args.mimic3_static_columns, horizon=args.horizons,
                                              scaling_method=args.scaler, clip_p=args.clipping)

    if args.command in ['train', 'evaluate']:
        load_weights = args.command == 'evaluate'
        reproducible = str(args.reproducible) == 'True'
        if not isinstance(args.seed, list):
            seeds = [args.seed]
        else:
            seeds = args.seed
        if not load_weights:
            gin_bindings, log_dir = get_bindings_and_params(args)
        else:
            gin_bindings, _ = get_bindings_and_params(args)
            log_dir = args.logdir
        if args.rs:
            reproducible = False
            max_attempt = 0
            is_already_ran = os.path.isdir(log_dir)
            while is_already_ran and max_attempt < 500:
                time.sleep(0.01 * np.random.randint(0, 10))
                gin_bindings, log_dir = get_bindings_and_params(args)
                is_already_ran = os.path.isdir(log_dir)
                max_attempt += 1
            if max_attempt >= 500:
                raise Exception('Reached max attempt to find unexplored set of parameters parameters')

        if args.task is not None:
            if isinstance(args.task, list) and (len(args.task) == args.num_class):
                gin_bindings_task = gin_bindings + [
                    'TASK = ' + str(args.task)]
                log_dir_task = os.path.join(log_dir, '_'.join(args.task))
                for seed in seeds:
                    if not load_weights:
                        log_dir_seed = os.path.join(log_dir_task, str(seed))
                    else:
                        log_dir_seed = log_dir_task
                    train_with_gin(model_dir=log_dir_seed,
                                   overwrite=args.overwrite,
                                   load_weights=load_weights,
                                   gin_config_files=args.config,
                                   gin_bindings=gin_bindings_task,
                                   seed=seed, reproducible=reproducible)
            else:
                for task in args.task:
                    gin_bindings_task = gin_bindings + [
                        'TASK = ' + "'" + str(task) + "'"]
                    log_dir_task = os.path.join(log_dir, str(task))
                    for seed in seeds:
                        if not load_weights:
                            log_dir_seed = os.path.join(log_dir_task, str(seed))
                        else:
                            log_dir_seed = log_dir_task
                        train_with_gin(model_dir=log_dir_seed,
                                       overwrite=args.overwrite,
                                       load_weights=load_weights,
                                       gin_config_files=args.config,
                                       gin_bindings=gin_bindings_task,
                                       seed=seed, reproducible=reproducible)
        else:
            for seed in seeds:
                if not load_weights:
                    log_dir_seed = os.path.join(log_dir, str(seed))
                train_with_gin(model_dir=log_dir_seed,
                               overwrite=args.overwrite,
                               load_weights=load_weights,
                               gin_config_files=args.config,
                               gin_bindings=gin_bindings, seed=seed, reproducible=reproducible)


"""Main module."""

if __name__ == '__main__':
    main()
