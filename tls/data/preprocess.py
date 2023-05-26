import gc
import logging
import os
import random

import numpy as np
import pandas as pd
import tables
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from tqdm import tqdm

from tls.common import constants


# HiRID
def gather_cat_values(common_path, cat_values):
    # not too many, so read all of them
    df_cat = pd.read_parquet(common_path, columns=list(cat_values))

    d = {}
    for c in df_cat.columns:
        d[c] = [x for x in df_cat[c].unique() if not np.isnan(x)]
    return d


def gather_stats_over_dataset(parts, to_standard_scale, to_min_max_scale, train_split_pids, fill_string,
                              clipping_quantile=0):
    minmax_scaler = MinMaxScaler()
    bottom_quantile, top_quantile = None, None
    counts = np.zeros(len(to_standard_scale) + len(to_min_max_scale))
    if clipping_quantile > 0:
        assert clipping_quantile < 0.5

        logging.info('Stats: Counting elemets')
        # we first collect counts
        for p in parts:
            df_part = pd.read_parquet(p, engine='pyarrow', columns=to_min_max_scale + to_standard_scale,
                                      filters=[(constants.PID, "in", train_split_pids)])
            df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            counts += df_part.notnull().sum().values
            gc.collect()

        quantile_width = (counts * clipping_quantile).astype(int)

        largest_quantile = quantile_width.max()
        top_values = - np.ones((largest_quantile, len(to_standard_scale) + len(to_min_max_scale))) * np.inf
        bottom_values = np.ones((largest_quantile, len(to_standard_scale) + len(to_min_max_scale))) * np.inf

        logging.info('Stats: Finding quantiles')
        # we gather top-quantile_width values for each columns
        for p in parts:
            df_part = pd.read_parquet(p, engine='pyarrow', columns=to_min_max_scale + to_standard_scale,
                                      filters=[(constants.PID, "in", train_split_pids)])
            df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            top_values = np.concatenate([df_part.replace(np.nan, -np.inf).values, top_values], axis=0)
            top_values = - np.partition(- top_values, largest_quantile, axis=0)[:largest_quantile]
            bottom_values = np.concatenate([df_part.replace(np.nan, np.inf).values, bottom_values], axis=0)
            bottom_values = np.partition(bottom_values, largest_quantile, axis=0)[:largest_quantile]
            gc.collect()

        top_quantile = - np.sort(- top_values, axis=0)[
            np.clip(quantile_width - 1, 0, np.inf).astype(int), np.arange(
                len(to_standard_scale) + len(to_min_max_scale))]
        bottom_quantile = np.sort(bottom_values, axis=0)[
            np.clip(quantile_width - 1, 0, np.inf).astype(int), np.arange(
                len(to_standard_scale) + len(to_min_max_scale))]

        # If no record for the measure we set quantiles to max values -inf, +inf
        top_quantile[np.where(top_quantile == - np.inf)] = np.inf
        bottom_quantile[np.where(bottom_quantile == np.inf)] = - np.inf

    logging.info('Stats: Finding Min-Max')
    for p in parts:
        df_part = impute_df(pd.read_parquet(p, engine='pyarrow', columns=to_min_max_scale + [constants.PID],
                                            filters=[(constants.PID, "in", train_split_pids)]), fill_string=fill_string)
        df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        if clipping_quantile > 0:
            df_part[to_min_max_scale] = df_part[to_min_max_scale].clip(bottom_quantile[:len(to_min_max_scale)],
                                                                       top_quantile[:len(to_min_max_scale)])
        minmax_scaler.partial_fit(df_part[to_min_max_scale])
        gc.collect()

    means = []
    stds = []
    # cannot read all to_standard_scale columns in memory, one-by-one would very slow, so read a certain number
    # of columns at a time
    if len(to_standard_scale) > 0:
        logging.info('Stats: Finding Mean-Std')
        batch_size = 20
        batches = (to_standard_scale[pos:pos + batch_size] for pos in range(0, len(to_standard_scale), batch_size))
        pos_to_scale = 0
        for s in batches:
            dfs = impute_df(pd.read_parquet(parts[0].parent, engine='pyarrow', columns=[constants.PID] + s,
                                            filters=[(constants.PID, "in", train_split_pids)]), fill_string=fill_string)
            dfs = dfs.replace(np.inf, np.nan).replace(-np.inf, np.nan)
            if clipping_quantile > 0:
                bot_batch_quantile = bottom_quantile[-len(to_standard_scale):][pos_to_scale:pos_to_scale + batch_size]
                top_batch_quantile = top_quantile[-len(to_standard_scale):][pos_to_scale:pos_to_scale + batch_size]
                dfs[s] = dfs[s].clip(bot_batch_quantile, top_batch_quantile)
                pos_to_scale += batch_size

            # don't rely on sklearn StandardScaler as partial_fit does not seem to work correctly
            # if in one iteration all values of a column are nan (i.e. the then mean becomes nan)
            means.extend(dfs[s].mean())
            stds.extend(dfs[s].std(ddof=0))  # ddof=0 to be consistent with sklearn StandardScalar
            gc.collect()

    # When there is only one measurement we set std to 1 so we don't divide by 0 and just center value.
    stds = np.array(stds)
    stds[np.where(stds == 0.0)] = 1.0
    stds = list(stds)
    return (means, stds), (bottom_quantile, top_quantile), minmax_scaler


def _normalize_cols(df, output_cols):
    cols_to_drop = [c for c in set(df.columns).difference(output_cols) if c != constants.PID]
    if cols_to_drop:
        logging.warning(f"Dropping columns {cols_to_drop} as they don't appear in output columns")
    df = df.drop(columns=cols_to_drop)

    cols_to_add = sorted(set(output_cols).difference(df.columns))

    if cols_to_add:
        logging.warning(f"Adding dummy columns {cols_to_add}")
        df[cols_to_add] = 0.0

    col_order = [constants.DATETIME] + sorted([c for c in df.columns if c != constants.DATETIME])
    df = df[col_order]

    cmp_list = list(c for c in df.columns if c != constants.PID)
    assert cmp_list == output_cols

    return df


def to_ml(save_path, parts, labels, features, endpoint_names, df_var_ref, fill_string, output_cols, split_path=None,
          random_seed=42, scaler='standard', clipping_quantile=0):
    df_part = pd.read_parquet(parts[0])
    data_cols = df_part.columns

    common_path = parts[0].parent
    df_pid_and_time = pd.read_parquet(common_path, columns=[constants.PID, constants.DATETIME])

    # list of patients for every split
    split_ids = get_splits(df_pid_and_time, split_path, random_seed)

    logging.info('Gathering variable types')
    cat_values, binary_values, to_standard_scale, to_min_max_scale = get_var_types(data_cols, df_var_ref, scaler)
    to_standard_scale = [c for c in to_standard_scale if c in set(output_cols)]
    to_min_max_scale = [c for c in to_min_max_scale if c in set(output_cols)]

    logging.info('Gathering categorical variables possible values')
    cat_vars_levels = gather_cat_values(common_path, cat_values)

    logging.info('Gathering stats for scaling')
    (means, stds), (bot_quant, top_quant), minmax_scaler = gather_stats_over_dataset(parts, to_standard_scale,
                                                                                     to_min_max_scale,
                                                                                     split_ids['train'],
                                                                                     fill_string, clipping_quantile)

    # for every train, val, test split keep how many records
    # have already been written (needed to compute correct window position)
    output_offsets = {}

    features_available = features
    if not features_available:
        features = [None] * len(parts)

    logging.info('Pre-processing per batch')
    for p, l, f in tqdm(zip(parts, labels, features)):
        df = impute_df(pd.read_parquet(p), fill_string=fill_string)

        # split features between historical feature and prsence features
        df_feat = pd.read_parquet(f) if f else pd.DataFrame(columns=[constants.PID])
        feat_names = df_feat.columns
        history_features_name = [name for name in feat_names if name.split('_')[0] != 'presence']
        presence_features_name = [name for name in feat_names if name.split('_')[0] == 'presence']
        presence_features_name = [constants.PID, constants.DATETIME] + presence_features_name
        presence_available = len(presence_features_name) > 2
        if presence_available:
            df_presence = df_feat[presence_features_name]
        else:
            df_presence = pd.DataFrame(columns=[constants.PID])
        if features_available:  # We extracted some features
            df_feat = df_feat[history_features_name]

        df_label = pd.read_parquet(l)[
            [constants.PID, constants.REL_DATETIME] + list(endpoint_names)]
        df_label = df_label.rename(columns={constants.REL_DATETIME: constants.DATETIME})
        df_label[constants.DATETIME] = df_label[constants.DATETIME] / 60.0

        # align indices between labels df and common df
        df_label = df_label.set_index([constants.PID, constants.DATETIME])
        df_label = df_label.reindex(index=zip(df[constants.PID].values, df[constants.DATETIME].values))
        df_label = df_label.reset_index()

        for cat_col in cat_values:
            df[cat_col] = pd.Categorical(df[cat_col], cat_vars_levels[cat_col])

        for bin_col in binary_values:
            bin_vals = [0.0, 1.0]
            if bin_col == 'sex':
                bin_vals = ['F', 'M']
            df[bin_col] = pd.Categorical(df[bin_col], bin_vals)

        if cat_values:
            df = pd.get_dummies(df, columns=cat_values)
        if binary_values:
            df = pd.get_dummies(df, columns=binary_values, drop_first=True)

        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        # reorder columns and making sure columns correspond to output_cols
        df = _normalize_cols(df, output_cols)

        split_dfs = {}
        split_labels = {}
        split_features = {}
        split_presence = {}
        for split in split_ids.keys():
            split_dfs[split] = df[df[constants.PID].isin(split_ids[split])]
            split_labels[split] = df_label[df_label[constants.PID].isin(split_ids[split])]
            split_features[split] = df_feat[df_feat[constants.PID].isin(split_ids[split])]
            split_presence[split] = df_presence[df_presence[constants.PID].isin(split_ids[split])]

        # windows computation: careful with offset!
        split_windows = {}
        for split, df in split_dfs.items():
            if df.empty:
                split_windows[split] = np.array([])
                continue
            split_windows[split] = get_windows_split(df, offset=output_offsets.get(split, 0))

            assert np.all(
                split_windows[split] == get_windows_split(split_labels[split], offset=output_offsets.get(split, 0)))
            split_dfs[split] = df.drop(columns=[constants.PID])
            split_labels[split] = split_labels[split].drop(columns=[constants.PID])
            split_features[split] = split_features[split].drop(columns=[constants.PID, constants.DATETIME])
            split_presence[split] = split_presence[split].drop(columns=[constants.PID, constants.DATETIME])

            output_offsets[split] = output_offsets.get(split, 0) + len(df)

        for split_df in split_dfs.values():
            if split_df.empty:
                continue

            if len(to_standard_scale) > 0:
                if clipping_quantile > 0:
                    split_df[to_standard_scale] = split_df[to_standard_scale].clip(bot_quant[-len(to_standard_scale):],
                                                                                   top_quant[-len(to_standard_scale):])
                split_df[to_standard_scale] = (split_df[to_standard_scale].values - means) / stds

            if clipping_quantile > 0:
                split_df[to_min_max_scale] = split_df[to_min_max_scale].clip(bot_quant[:len(to_min_max_scale)],
                                                                             top_quant[:len(to_min_max_scale)])

            split_df[to_min_max_scale] = minmax_scaler.transform(split_df[to_min_max_scale])
            split_df.replace(np.inf, np.nan, inplace=True)
            split_df.replace(-np.inf, np.nan, inplace=True)

        split_arrays = {}
        label_arrays = {}
        feature_arrays = {}
        presence_arrays = {}
        for split, df in split_dfs.items():
            array_split = df.values
            array_label = split_labels[split].values

            np.place(array_split, mask=np.isnan(array_split), vals=0.0)

            split_arrays[split] = array_split
            label_arrays[split] = array_label

            if features_available:
                array_features = split_features[split].values
                np.place(array_features, mask=np.isnan(array_features), vals=0.0)
                feature_arrays[split] = array_features

            if presence_available:
                array_presence = split_presence[split].values
                np.place(array_presence, mask=np.isnan(array_presence), vals=0.0)
                presence_arrays[split] = array_presence

            assert len(df.columns) == split_arrays[split].shape[1]

        tasks = list(split_labels['train'].columns)

        output_cols = [c for c in df.columns if c != constants.PID]

        feature_names = list(split_features['train'].columns)
        presence_names = list(split_presence['train'].columns)

        save_to_h5_with_tasks(save_path, output_cols, tasks, feature_names,
                              presence_names, split_arrays, label_arrays,
                              feature_arrays if features_available else None,
                              presence_arrays if presence_available else None,
                              split_windows)

        gc.collect()


def get_utf8_cols(col_array):
    return np.array([t.decode('utf-8') for t in col_array[:]])


def merge_multiple_horizon_labels(save_path, tables_path, label_radicals, horizons, joint_table_idx=0):
    horizon_cols = {'train': [],
                    'test': [],
                    'val': []}
    h5_tables = [tables.open_file(data_path, "r").root for data_path in tables_path]
    columns = []
    source_joint_table = h5_tables[joint_table_idx]
    all_labels = get_utf8_cols(source_joint_table['labels']['tasks'])
    hor_labels = [rad + '_' + str(horizons[joint_table_idx]) + 'Hours' for rad in label_radicals]
    other_labels_idx = [i for i, k in enumerate(all_labels) if k not in hor_labels]
    other_labels_name = all_labels[other_labels_idx]
    columns += list(other_labels_name)

    for split in horizon_cols.keys():
        horizon_cols[split].append(source_joint_table['labels'][split][:, other_labels_idx])

    for table, horizon in zip(h5_tables, horizons):
        all_labels = get_utf8_cols(table['labels']['tasks'])
        labels_name = []
        labels_idx = []
        for radical in label_radicals:
            label_name = radical + '_' + str(horizon) + 'Hours'
            label_idx = np.where(all_labels == label_name)[0][0]
            labels_name.append(label_name)
            labels_idx.append(label_idx)

        for split in horizon_cols.keys():
            horizon_cols[split].append(table['labels'][split][:, labels_idx])

        columns += labels_name
    for split in horizon_cols.keys():
        horizon_cols[split] = np.concatenate(horizon_cols[split], axis=-1)

    col_names = get_utf8_cols(source_joint_table['data']['columns'])
    task_names = columns
    feature_names = get_utf8_cols(source_joint_table['features']['name_features'])
    presence_names = get_utf8_cols(source_joint_table['presence_features']['name_features'])

    data_dict = {k: source_joint_table['data'][k][:] for k in ['train', 'test', 'val']}
    features_dict = {k: source_joint_table['features'][k][:] for k in ['train', 'test', 'val']}
    presence_dict = {k: source_joint_table['presence_features'][k][:] for k in ['train', 'test', 'val']}

    patient_windows_dict = {k: source_joint_table['patient_windows'][k][:] for k in ['train', 'test', 'val']}
    label_dict = horizon_cols
    save_to_h5_with_tasks(save_path, col_names, task_names, feature_names, presence_names,
                          data_dict, label_dict, features_dict, presence_dict, patient_windows_dict)


def impute_df(df, fill_string='ffill'):
    if fill_string is not None:
        df = df.groupby(constants.PID).apply(lambda x: x.fillna(method=fill_string))
    return df


def get_var_types(columns, df_var_ref, scaler='standard'):
    cat_ref = list(df_var_ref[df_var_ref.variableunit == 'Categorical']['metavariablename'].values)
    cat_values = [c for c in cat_ref if c in columns]
    binary_values = list(np.unique(df_var_ref[df_var_ref['metavariableunit'] == 'Binary']['metavariablename']))
    binary_values += ['sex']
    to_min_max_scale = [constants.DATETIME, 'admissiontime']
    if scaler == 'standard':
        to_standard_scale = [k for k in np.unique(df_var_ref['metavariablename'].astype(str)) if
                             k not in cat_values + binary_values] + ['age', 'height']
        to_standard_scale = [c for c in to_standard_scale if c in columns]
    elif scaler == 'minmax':
        to_standard_scale = []
        to_min_max_scale += [k for k in np.unique(df_var_ref['metavariablename'].astype(str)) if
                             k not in cat_values + binary_values] + ['age', 'height']
        to_min_max_scale = [c for c in to_min_max_scale if c in columns]
    else:
        raise Exception('scaler has to be standard or minmax')

    return cat_values, binary_values, to_standard_scale, to_min_max_scale


def get_splits(df, split_path, random_seed):
    if split_path:
        split_df = pd.read_csv(split_path, sep='\t')
        split_ids = {}
        for split in split_df['split'].unique():
            split_ids[split] = split_df.loc[split_df['split'] == split, constants.PID].values
    else:
        split_ids = {}
        train_val_ids, split_ids['test'] = train_test_split(np.unique(df[constants.PID]), test_size=0.15,
                                                            random_state=random_seed)
        split_ids['train'], split_ids['val'] = train_test_split(train_val_ids, test_size=(0.15 / 0.85),
                                                                random_state=random_seed)
    return split_ids


def get_windows_split(df_split, offset=0):
    pid_array = df_split[constants.PID]
    starts = sorted(np.unique(pid_array, return_index=True)[1])
    stops = np.concatenate([starts[1:], [df_split.shape[0]]])
    ids = pid_array.values[starts]
    return np.stack([np.array(starts) + offset, np.array(stops) + offset, ids], axis=1)


# MIMIC-III

def impute_sample_1h(measures, stay_time=None, time_idx=0):
    """Forward imputes with 1h resolution any stay to duration stay_time.
    Time has to be at index_0
    Args:
        measures: Array-like matrix with successive measurement.
        stay_time: (Optional) Time until which we want to impute.
    Returns:
        Imputed time-series.
    """
    forward_filled_sample = impute_sample(measures)
    imputed_sample = [np.array(forward_filled_sample[0])]
    imputed_sample[0][time_idx] = 0
    if not stay_time:
        max_time = int(np.ceil(float(measures[-1, time_idx])))
    else:
        max_time = int(np.ceil(stay_time))
    for k in range(1, max_time + 1):
        diff_to_k = forward_filled_sample[:, time_idx].astype(float) - k
        if np.argwhere(diff_to_k <= 0).shape[0] > 0:
            idx_for_k = np.argwhere(diff_to_k <= 0)[-1][0]
            time_k = np.array(forward_filled_sample[idx_for_k])
            time_k[time_idx] = k
            imputed_sample.append(time_k)
        else:
            time_k = np.array(imputed_sample[-1])
            time_k[time_idx] = k
            imputed_sample.append(time_k)
    imputed_sample = np.stack(imputed_sample, axis=0)

    return imputed_sample


def impute_sample(measures_t):
    """ Used under impute_sample_1h to forward impute without re-defining the resolution.
    """
    measures = np.array(measures_t)
    imputed_sample = [measures[0]]
    for k in range(1, len(measures)):
        r_t = measures[k]
        r_t_m_1 = np.array(imputed_sample[-1])
        idx_to_impute = np.argwhere(r_t == '')
        r_t[idx_to_impute] = r_t_m_1[idx_to_impute]
        imputed_sample.append(np.array(r_t))
    imputed_sample = np.stack(imputed_sample, axis=0)
    return imputed_sample


def remove_strings_col(data, col, channel_to_id, matching_dict):
    """Replaces the string arguments existing in the MIMIC-III data to category index.
    """
    transfo_data = {}
    for split in ['train', 'test', 'val']:
        current_data = np.copy(data[split])
        for channel in col:
            if channel in list(matching_dict.keys()):
                m_dict = matching_dict[channel]
                m_dict[''] = np.nan
                idx_channel = channel_to_id[channel]
                data_channel = current_data[:, idx_channel]
                r = list(map(lambda x: m_dict[x], data_channel))
                current_data[:, idx_channel] = np.array(r)
            else:
                idx_channel = channel_to_id[channel]
                data_channel = current_data[:, idx_channel]
                data_channel[np.where(data_channel == '')] = np.nan
                current_data[:, idx_channel] = data_channel.astype(float)
        transfo_data[split] = current_data.astype(float)
    return transfo_data


def extract_raw_data(base_path):
    """Wrapper around MultitaskReader to extract MIMIC-III benchmark data to our h5 format.
    Args:
        base_path: Path to source data 'data/multitask'.
        You obtain it with this command 'python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/'.
    Returns:
        data_d: Dict with data array concatenating all stays for each split.
        labels_d: Dict with labels array associate data_d for each split.
        patient_windows_d : Containing the (start, stop, patient_id) for each stay in each split.
        col: Name of the columns in data
        tasks: Name of the columns in labels
    """
    data_d = {}
    labels_d = {}
    patient_window_d = {}
    text_window_d = {}

    for split in ['train', 'test', 'val']:
        print('Extracting Data for {} Split.'.format(split))
        if split in ['train', 'val']:
            folder = os.path.join(base_path, 'train')
        else:
            folder = os.path.join(base_path, 'test')

        file = os.path.join(base_path, split + '_listfile.csv')
        sample_reader = MultitaskReader(folder, file)
        num_samples = sample_reader.get_number_of_examples()
        lookup_table = []
        td_start_stop_id = []
        start_stop_id = []
        labels_split = []
        current_idx = 0

        for idx in tqdm(range(num_samples)):
            patient_sample = sample_reader.read_example(idx)
            col = list(patient_sample['header'])
            d = patient_sample['X']
            imputed_d = impute_sample_1h(d, float(patient_sample['t']))
            patient_id = int(patient_sample['name'].split('_')[0])
            episode_nb = int(patient_sample['name'].split('_')[1][-1])
            stay_id = episode_nb * 10000000 + patient_id  # We avoid confusing different episodes
            label_los = patient_sample['los']
            label_decomp = patient_sample['decomp']
            label_ihm = patient_sample['ihm']
            label_pheno = patient_sample['pheno']

            n_step = int(np.ceil(patient_sample['t']))

            # Handling of samples where LOS and Decomp masks are not same shape
            if len(patient_sample['los'][0]) > n_step:
                label_los = (patient_sample['los'][0][:n_step], patient_sample['los'][1][:n_step])
            elif len(patient_sample['los'][0]) < n_step:
                raise Exception()
            if len(patient_sample['decomp'][0]) > n_step:
                label_decomp = (patient_sample['decomp'][0][:n_step], patient_sample['decomp'][1][:n_step])
            if len(label_decomp[0]) - len(label_los[0]) < 0:
                adding_mask = [0 for k in range(abs(len(label_decomp[0]) - len(label_los[0])))]
                adding_label = [-1 for k in range(abs(len(label_decomp[0]) - len(label_los[0])))]
                new_mask = label_decomp[0] + adding_mask
                new_labels = label_decomp[1] + adding_label
                label_decomp = (new_mask, new_labels)
                assert len(label_decomp[0]) - len(label_los[0]) == 0
            elif len(label_decomp[0]) - len(label_los[0]):
                raise Exception()

            # We build labels in our format with np.nan when we don't have a label
            mask_decomp, label_decomp = label_decomp
            mask_los, label_los = label_los
            mask_decomp = np.array(mask_decomp).astype(float)
            mask_los = np.array(mask_los).astype(float)
            mask_decomp[np.argwhere(mask_decomp == 0)] = np.nan
            mask_los[np.argwhere(mask_los == 0)] = np.nan
            masked_labels_los = np.concatenate([[np.nan], mask_los * np.array(label_los)], axis=0)
            masked_labels_decomp = np.concatenate([[np.nan], mask_decomp * np.array(label_decomp).astype(float)],
                                                  axis=0)
            masked_labels_ihm = masked_labels_los
            masked_labels_ihm[:] = np.nan
            masked_labels_pheno = np.tile(masked_labels_ihm.reshape(-1, 1), (1, len(label_pheno)))
            if label_ihm[1] == 1:
                masked_labels_ihm[label_ihm[0]] = label_ihm[2]
            masked_labels_pheno[-1] = label_pheno

            assert imputed_d.shape[0] == masked_labels_los.shape[-1]

            # Data
            lookup_table.append(imputed_d)
            start_stop_id.append([current_idx, current_idx + len(imputed_d), stay_id])
            current_idx = current_idx + len(imputed_d)

            labels_split.append(np.concatenate([masked_labels_los.reshape(1, -1), masked_labels_decomp.reshape(1, -1),
                                                masked_labels_ihm.reshape(1, -1), masked_labels_pheno.T], axis=0))

        data_d[split] = np.concatenate(lookup_table, axis=0)

        labels_d[split] = np.concatenate(labels_split, axis=1).T
        patient_window_d[split] = np.array(start_stop_id)
        text_window_d[split] = np.array(td_start_stop_id)
        col = list(patient_sample['header'])
        tasks = ['los', 'decomp', 'ihm'] + ['pheno_' + str(k) for k in range(len(label_pheno))]

    return data_d, labels_d, patient_window_d, col, tasks


def put_static_first(data, col, static_col):
    """Simple function putting the static columns first in the data.
    Args:
        data: Dict with a data array for each split.
        col: Ordered list of the columns in the data.
        static_col: List of static columns names.
    Returns:
        data_inverted : Analog  to data with columns reordered in each split.
        col_inverted : Analog to col woth columns names reordered.
    """
    static_index = list(np.where(np.isin(np.array(col), static_col))[0])
    n_col = len(col)
    non_static_index = [k for k in range(n_col) if k not in static_index]
    new_idx = static_index + non_static_index
    data_inverted = {}
    for split in data.keys():
        data_inverted[split] = data[split][:, new_idx]
    col_inverted = list(np.array(col)[new_idx])
    return data_inverted, col_inverted


def clip_dataset(var_range, data, columns):
    """Set each values outside of predefined range to NaN.
    Args:
        var_range: Dict with associated range [min,max] to each variable name.
        data: Dict with a data array for each split.
        columns: Ordered list of the columns in the data.
    Returns:
        new_data : Data with no longer any value outside of the range.
    """
    new_data = {}
    for split in ['train', 'test', 'val']:
        clipped_data = data[split][:]
        for i, col in enumerate(columns):
            if var_range.get(col):
                idx = np.sort(np.concatenate([np.argwhere(clipped_data[:, i] > var_range[col][1]),
                                              np.argwhere(clipped_data[:, i] < var_range[col][0])])[:, 0])
                clipped_data[idx, i] = np.nan
        new_data[split] = clipped_data
    return new_data


def finding_cat_features(rep_data, threshold):
    """
    Extracts the index and names of categorical in a pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        threshold: Number of uniqur value below which we consider a variable as categorical if it's an integer
    Returns:
        categorical: List of names containing categorical features.
        categorical_idx: List of matching column indexes.
    """
    columns = np.array([name.decode('utf-8') for name in rep_data['data']['columns'][:]])

    categorical = []

    for i, c in enumerate(columns):
        values = rep_data['data']['train'][:, i]
        values = values[~np.isnan(values)]
        nb_values = len(np.unique(values))

        if nb_values <= threshold and np.all(values == values.astype(int)):
            categorical.append(c)

    categorical_idx = np.sort([np.argwhere(columns == feat)[0, 0] for feat in categorical])

    return categorical, categorical_idx


def finding_cat_features_fom_file(rep_data, info_df):
    """
    Extracts the index and names of categorical in a pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        info_df: Dataframe with information on each variable.
    Returns:
        categorical: List of names containing categorical features.
        categorical_idx: List of matching column indexes.
    """
    columns = np.array([name.decode('utf-8') for name in rep_data['data']['columns'][:]])
    categorical = []

    for i, c in enumerate(columns):
        if c.split('_')[0] != 'plain':
            pass
        else:
            if info_df[info_df['VariableID'] == c.split('_')[-1]]['Datatype'].values == 'Categorical':
                categorical.append(c)
    categorical_idx = np.sort([np.argwhere(columns == feat)[0, 0] for feat in categorical])
    return categorical, categorical_idx


def get_one_hot(rep_data, cat_names, cat_idx):
    """
    One-hots the categorical features in a given pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        cat_names: List of names containing categorical features.
        cat_idx: List of matching column indexes.
    Returns:
        all_categorical_data: Dict with each split one-hotted categorical column as a big array.
        col_name: List of name of the matching columns
    """
    all_categorical_data = np.concatenate([rep_data['data']['train'][:, cat_idx],
                                           rep_data['data']['test'][:, cat_idx],
                                           rep_data['data']['val'][:, cat_idx]], axis=0)
    cat_dict = {}
    col_name = []
    for i, cat in enumerate(cat_idx):
        dum = np.array(pd.get_dummies(all_categorical_data[:, i]))
        if dum.shape[-1] <= 2:
            dum = dum[:, -1:]
            col_name += [cat_names[i].split('_')[-1] + '_cat']
        else:
            col_name += [cat_names[i].split('_')[-1] + '_cat_' + str(k) for k in range(dum.shape[-1])]
        cat_dict[cat] = dum

    all_categorical_data_one_h = np.concatenate(list(cat_dict.values()), axis=1)

    all_categorical_data = {}
    all_categorical_data['train'] = all_categorical_data_one_h[:rep_data['data']['train'].shape[0]]
    all_categorical_data['test'] = all_categorical_data_one_h[
                                   rep_data['data']['train'].shape[0]:rep_data['data']['train'].shape[0] +
                                                                      rep_data['data']['test'].shape[0]]
    all_categorical_data['val'] = all_categorical_data_one_h[-rep_data['data']['val'].shape[0]:]

    return all_categorical_data, col_name


def extend_labels_decomp(labels, patient_windows, horizons, true_horizon):
    """
    After checking individually MIMIC-III patient the correct edge in ambiguous cases is the Decomp one not LOS.
    Thus we define event date based on this eventhough sometimes "stays" can theoritically carry on
    even though the patient is deceased.

    Returns:
        columns_name: list of strings with columns name
        new_labels: array with new labels in the same order as columns_name.
    """
    new_labels = {}
    event_windows = []

    # Define events
    for start, stop, _ in patient_windows:
        p_label = labels[start:stop]
        if np.any(p_label == 1):
            pos = np.where(p_label == 1)[0]
            edge_l, edge_r, = pos[0], pos[-1]

            # We need to check which edge is correctly aligned with death
            # This is because some patient died outside of the ICU
            assert edge_r - edge_l <= true_horizon

            if p_label[edge_l - 1] == 0:
                true_l = True
            elif np.isnan(p_label[edge_l - 1]) and edge_r - edge_l == true_horizon:
                true_l = True
            else:
                true_l = False

            if true_l and edge_r - edge_l <= true_horizon:
                true_r = False
            else:
                true_r = True

            if true_r:
                event = edge_r + start + 1
            elif true_l:
                event = edge_l + start + true_horizon
            else:
                raise Exception('One of the two edges has to be true')
            event_windows.append([start, stop, event])
        else:
            event_windows.append([start, stop, -1])
    event_windows = np.array(event_windows)

    new_event_labels = np.zeros(labels.shape)
    new_events_idx = []
    new_events_value = []

    for start, stop, event_idx in event_windows:
        if event_idx != -1:  # positive case
            diff_to_end = stop - 1 - event_idx
            if diff_to_end < 0:
                actual_idx = stop - 1
                value = - diff_to_end + 1
            else:
                actual_idx = event_idx
                value = 1
            new_events_idx.append(actual_idx)
            new_events_value.append(value)

    new_events_idx = np.array(new_events_idx)
    new_events_value = np.array(new_events_value)
    new_event_labels[new_events_idx] = new_events_value
    new_labels['Event'] = new_event_labels

    for new_h in horizons:
        new_h_labels = np.zeros(labels.shape)
        new_labels_idx = []
        for start, stop, event in event_windows:
            if event != -1:
                new_candidates = [event - k for k in range(1, new_h + 1)]
                new_indexes = np.array([k for k in new_candidates if ((k < stop) and (k >= start))])
                assert len(new_indexes) <= new_h
                assert np.all(new_indexes.astype(int) == new_indexes)
                new_labels_idx.append(new_indexes.astype(int))

        new_labels_idx = np.concatenate(new_labels_idx)
        new_h_labels[new_labels_idx] = 1
        new_h_labels[np.where(np.isnan(labels))] = np.nan
        new_labels[str(new_h) + 'Hours'] = new_h_labels

    return np.stack(new_labels.values(), axis=1), ['decomp_{}'.format(k) for k in new_labels.keys()]


def get_labels_with_new_decomp_horizons(labels, patient_windows, tasks, horizons, true_horizon=24):
    """
    Wraps around extend_labels_decomp to extract additional horizons for decomp
    """

    decomp_idx = np.where(np.array(tasks) == 'decomp')[0][0]
    new_labels = {}
    for split in ['train', 'test', 'val']:
        labels_split = labels[split][:, decomp_idx]
        patient_windows_split = patient_windows[split][:]
        new_split_labels, new_tasks = extend_labels_decomp(labels_split, patient_windows_split, horizons, true_horizon)
        new_split_labels = np.concatenate([labels[split][:], new_split_labels], axis=-1)
        new_labels[split] = new_split_labels
    full_tasks = np.concatenate([tasks, np.array(new_tasks)])

    return new_labels, full_tasks


def scaling_data_common(data_path, scaled_path, threshold=25, scaling_method='standard', scaling_kwargs={},
                        static_idx=None, df_ref=None, clip_p=0):
    """
    Wrapper which one-hot and scales the a pre-built dataset.
    Args:
        data_path: String with the path to the pre-built non scaled dataset
        threshold: Int below which we consider a variable as categorical
        scaling_method: String with the scaling to apply.
        static_idx: List of indexes containing static columns.
        df_ref: Reference dataset containing supplementary information on the columns.
    Returns:
        data_dic: dict with each split as a big array.
        label_dic: dict with each split and and labels array in same order as lookup_table.
        patient_dic: dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
        col: list of the variables names corresponding to each column.
        labels_name: list of the tasks name corresponding to labels columns.
    """
    if scaling_method == 'standard':
        scaler = StandardScaler(**scaling_kwargs)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler(**scaling_kwargs)
    elif scaling_method == 'yeo':
        scaler = PowerTransformer(**scaling_kwargs)
    elif scaling_method == 'robust':
        scaler = RobustScaler(**scaling_kwargs)
    elif scaling_method == 'quantile':
        scaling_kwargs['subsample'] = int(1e6)
        scaling_kwargs['n_quantiles'] = 256
        scaler = QuantileTransformer(**scaling_kwargs)
    elif scaling_method == 'quantile-gaussian':
        scaling_kwargs['subsample'] = int(1e6)
        scaling_kwargs['n_quantiles'] = 256
        scaling_kwargs['output_distribution'] = 'normal'
        scaler = QuantileTransformer(**scaling_kwargs)
    else:
        raise Exception("scaling method needs to be: 'standard', 'minmax', 'yeo', or 'quantile'")
    rep_data = tables.open_file(data_path, "r").root
    columns = np.array([name.decode('utf-8') for name in rep_data['data']['columns'][:]])
    train_data = rep_data['data']['train'][:]
    test_data = rep_data['data']['test'][:]
    val_data = rep_data['data']['val'][:]

    time_idx = np.where(columns == 'Hours')[0]  # We scale time with a MinMaxScaler

    # We just extract tasks name to propagate
    if rep_data.__contains__('labels'):
        labels_name = np.array([name.decode('utf-8') for name in rep_data['labels']['tasks'][:]])
    else:
        labels_name = None
    # We treat np.inf and np.nan as the same
    np.place(train_data, mask=np.isinf(train_data), vals=np.nan)
    np.place(test_data, mask=np.isinf(test_data), vals=np.nan)
    np.place(val_data, mask=np.isinf(val_data), vals=np.nan)

    if clip_p > 0:
        low = np.nanpercentile(train_data, clip_p, 0)
        high = np.nanpercentile(train_data, 100 - clip_p, 0)
        train_data_scaled = np.clip(train_data, low, high)
        val_data_scaled = np.clip(val_data, low, high)
        test_data_scaled = np.clip(test_data, low, high)
    else:
        train_data_scaled = train_data
        val_data_scaled = val_data
        test_data_scaled = test_data

    train_data_scaled = scaler.fit_transform(train_data_scaled)
    val_data_scaled = scaler.transform(val_data_scaled)
    test_data_scaled = scaler.transform(test_data_scaled)

    # We pad after scaling, Thus zero is equivalent to padding with the mean value across patient
    np.place(train_data_scaled, mask=np.isnan(train_data_scaled), vals=0.0)
    np.place(test_data_scaled, mask=np.isnan(test_data_scaled), vals=0.0)
    np.place(val_data_scaled, mask=np.isnan(val_data_scaled), vals=0.0)

    # If we have static values we take one per patient stay
    if static_idx:
        train_static_values = train_data[rep_data['patient_windows']['train'][:][:, 0]][:, static_idx]
        static_scaler = StandardScaler()
        static_scaler.fit(train_static_values)

        # Scale all entries
        train_data_static_scaled = static_scaler.transform(train_data[:, static_idx])
        val_data_static_scaled = static_scaler.transform(val_data[:, static_idx])
        test_data_static_scaled = static_scaler.transform(test_data[:, static_idx])
        # Replace NaNs
        np.place(train_data_static_scaled, mask=np.isnan(train_data_static_scaled), vals=0.0)
        np.place(val_data_static_scaled, mask=np.isnan(val_data_static_scaled), vals=0.0)
        np.place(test_data_static_scaled, mask=np.isnan(test_data_static_scaled), vals=0.0)

        # Insert in the scaled dataset
        train_data_scaled[:, static_idx] = train_data_static_scaled
        test_data_scaled[:, static_idx] = test_data_static_scaled
        val_data_scaled[:, static_idx] = val_data_static_scaled

    if time_idx:
        train_time_values = train_data[:, time_idx]
        time_scaler = MinMaxScaler()
        time_scaler.fit(train_time_values)

        # Scale all entries
        train_data_time_scaled = time_scaler.transform(train_data[:, time_idx])
        val_data_time_scaled = time_scaler.transform(val_data[:, time_idx])
        test_data_time_scaled = time_scaler.transform(test_data[:, time_idx])

        # Insert in the scaled dataset
        train_data_scaled[:, time_idx] = train_data_time_scaled
        test_data_scaled[:, time_idx] = test_data_time_scaled
        val_data_scaled[:, time_idx] = val_data_time_scaled

    # We deal with the categorical features.
    if df_ref is None:
        cat_names, cat_idx = finding_cat_features(rep_data, threshold)
    else:
        cat_names, cat_idx = finding_cat_features_fom_file(rep_data, df_ref)

    # We check for columns that are both categorical and static
    if static_idx:
        common_idx = [idx for idx in cat_idx if idx in static_idx]
        if common_idx:
            common_name = columns[common_idx]
        else:
            common_name = None

    if len(cat_names) > 0:
        # We one-hot categorical features with more than two possible values
        all_categorical_data, oh_cat_name = get_one_hot(rep_data, cat_names, cat_idx)
        if common_name is not None:
            common_cat_name = [c for c in oh_cat_name if c.split('_')[0] in common_name]

        # We replace them at the end of the features
        train_data_scaled = np.concatenate([np.delete(train_data_scaled, cat_idx, axis=1),
                                            all_categorical_data['train']], axis=-1)
        test_data_scaled = np.concatenate([np.delete(test_data_scaled, cat_idx, axis=1),
                                           all_categorical_data['test']], axis=-1)
        val_data_scaled = np.concatenate([np.delete(val_data_scaled, cat_idx, axis=1),
                                          all_categorical_data['val']], axis=-1)
        columns = np.concatenate([np.delete(columns, cat_idx, axis=0), oh_cat_name], axis=0)

        # We ensure that static categorical features are also among the first features with other static ones.
        if common_name is not None:
            common_current_idx = [i for i, n in enumerate(columns) if n in common_cat_name]
            new_idx = common_current_idx + [k for k in range(len(columns)) if k not in common_current_idx]
            columns = columns[new_idx]
            train_data_scaled = train_data_scaled[:, new_idx]
            test_data_scaled = test_data_scaled[:, new_idx]
            val_data_scaled = val_data_scaled[:, new_idx]

    data_dic = {'train': train_data_scaled,
                'test': test_data_scaled,
                'val': val_data_scaled}

    if rep_data.__contains__('labels'):
        label_dic = {split: rep_data['labels'][split][:] for split in data_dic.keys()}
    else:
        label_dic = None

    if rep_data.__contains__('patient_windows'):

        patient_dic = {split: rep_data['patient_windows'][split][:] for split in data_dic.keys()}
    else:
        patient_dic = None

    save_to_h5_with_tasks(scaled_path, col_names=columns, task_names=labels_name, feature_names=None,
                          presence_names=None, data_dict=data_dic, label_dict=label_dic,
                          patient_windows_dict=patient_dic, features_dict=None, presence_dict=None)

    return data_dic, label_dic, patient_dic, columns, labels_name, time_scaler


def run_non_scaled_pipe(base_path, non_scaled_path, channel_to_id, matching_dict, var_range, static_col=['Height'],
                        horizons=None):
    """Wrapper around the full pre-process
    """
    data_d, labels_d, patient_window_d, col, tasks = extract_raw_data(base_path)

    no_string_data = remove_strings_col(data_d, col, channel_to_id, matching_dict)

    clipped_data = clip_dataset(var_range, no_string_data, col)

    data_inverted, col_inverted = put_static_first(clipped_data, col, static_col)

    if horizons is not None:
        labels_d, tasks = get_labels_with_new_decomp_horizons(labels_d, patient_window_d, tasks, horizons)
    save_to_h5_with_tasks(non_scaled_path, col_names=col_inverted, task_names=tasks, feature_names=None,
                          presence_names=None, data_dict=data_inverted, label_dict=labels_d, features_dict=None,
                          patient_windows_dict=patient_window_d, presence_dict=None)

    return data_inverted, labels_d, patient_window_d, col_inverted, tasks


## Reader classes directly from https://github.com/YerevaNN/mimic3-benchmarks
class Reader(object):
    """Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.
    """

    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(float, x[len(x) // 2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(int, x[len(x) // 2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}


def _write_data_to_hdf(data, dataset_name, node, f, first_write, nr_cols, expectedrows=1000000):
    filters = tables.Filters(complevel=5, complib='blosc:lz4')

    if first_write:
        ea = f.create_earray(node, dataset_name,
                             atom=tables.Atom.from_dtype(data.dtype),
                             expectedrows=expectedrows,
                             shape=(0, nr_cols),
                             filters=filters)
        if len(data) > 0:
            ea.append(data)
    elif len(data) > 0:
        node[dataset_name].append(data)


def save_to_h5_with_tasks(save_path, col_names, task_names, feature_names, presence_names,
                          data_dict, label_dict, features_dict, presence_dict, patient_windows_dict):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        task_names: List of names for the tasks in the dataset.
        feature_names: List of names for the features in the dataset.
        presence_names: List of names for the presence features in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and labels array in same order as data_dict.
        features_dict: (Optional) Dict with each split and features array in same order as data_dict.
        presence_dict: (Optional) Dict with each split and presence features array in the same order as data_dict.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
    Returns:
    """

    # data labels windows

    first_write = not save_path.exists()
    mode = 'w' if first_write else 'a'

    with tables.open_file(save_path, mode) as f:
        if first_write:
            n_data = f.create_group("/", 'data', 'Dataset')
            f.create_array(n_data, 'columns', obj=[str(k).encode('utf-8') for k in col_names])
        else:
            n_data = f.get_node('/data')

        splits = ['train', 'val', 'test']
        for split in splits:
            _write_data_to_hdf(data_dict[split].astype(float), split, n_data, f, first_write,
                               data_dict['train'].shape[1])

        if label_dict is not None:
            if first_write:
                labels = f.create_group("/", 'labels', 'Labels')
                f.create_array(labels, 'tasks', obj=[str(k).encode('utf-8') for k in task_names])
            else:
                labels = f.get_node('/labels')

            for split in splits:
                _write_data_to_hdf(label_dict[split].astype(float), split, labels, f, first_write,
                                   label_dict['train'].shape[1])

        if features_dict is not None:
            if first_write:
                features = f.create_group("/", 'features', 'Features')
                f.create_array(features, 'name_features', obj=[str(k).encode('utf-8') for k in feature_names])
            else:
                features = f.get_node('/features')

            for split in splits:
                _write_data_to_hdf(features_dict[split].astype(float), split, features, f, first_write,
                                   features_dict['train'].shape[1])

        if presence_dict is not None:
            if first_write:
                presence_features = f.create_group("/", 'presence_features', 'Presence Features')
                f.create_array(presence_features, 'name_features', obj=[str(k).encode('utf-8') for k in presence_names])
            else:
                presence_features = f.get_node('/presence_features')

            for split in splits:
                _write_data_to_hdf(presence_dict[split].astype(float), split, presence_features, f, first_write,
                                   presence_dict['train'].shape[1])

        if patient_windows_dict is not None:
            if first_write:
                p_windows = f.create_group("/", 'patient_windows', 'Windows')
            else:
                p_windows = f.get_node('/patient_windows')

            for split in splits:
                _write_data_to_hdf(patient_windows_dict[split].astype(int), split, p_windows, f, first_write,
                                   patient_windows_dict['train'].shape[1])

        if not len(col_names) == data_dict['train'].shape[-1]:
            raise Exception(
                "We saved to data but the number of columns ({}) didn't match the number of features {} ".format(
                    len(col_names), data_dict['train'].shape[-1]))
