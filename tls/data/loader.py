import logging

import gin
import numpy as np
import tables
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from tls.data.utils import get_smoothed_labels, process_multihorizon_task_string


@gin.configurable('ICUVariableLengthDataset')
class ICUVariableLengthDataset(Dataset):
    """torch.Dataset built around ICUVariableLengthLoaderTables """

    def __init__(self, source_path, split='train', maxlen=-1, scale_label=False, scale_feature=False):
        """
        Args:
            source_path (string): Path to the source h5 file.
            split (string): Either 'train','val' or 'test'.
            maxlen (int): Max size of the generated sequence. If -1, takes the max size existing in split.
            scale_label (bool): Whether or not to train a min_max scaler on labels (For regression stability).
        """
        self.h5_loader = ICUVariableLengthLoaderTables(source_path, batch_size=1, maxlen=maxlen, splits=[split])
        self.split = split
        self.maxlen = self.h5_loader.maxlen
        self.scale_label = scale_label
        self.scale_feature = scale_feature
        self.smooth_labels = self.h5_loader.smooth
        if self.scale_label:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.get_labels().reshape(-1, 1))
        else:
            self.scaler = None

        if self.scale_feature and self.h5_loader.use_feat:
            self.feat_scaler = MinMaxScaler()
            if split == 'train':
                self.feat_scaler.fit(self.h5_loader.feature_table[split])
                self.h5_loader.feature_table[split] = self.feat_scaler.transform(self.h5_loader.feature_table[split])
        else:
            self.feat_scaler = None

    def __len__(self):
        return self.h5_loader.num_samples[self.split]

    def __getitem__(self, idx):
        data, pad_mask, label, event = self.h5_loader.sample(None, self.split, idx)

        if self.scale_label:
            label = self.scaler.transform(label.reshape(-1, 1))[:, 0]

        return data, label, pad_mask, event

    def set_scaler(self, scaler):
        """Sets the scaler for labels in case of regression.

        Args:
            scaler: sklearn scaler instance

        """
        self.scaler = scaler

    def set_feat_scaler(self, feat_scaler):
        """Sets the scaler for labels in case of regression.

        Args:
            feat_scaler: sklearn scaler instance

        """
        self.feat_scaler = feat_scaler
        self.h5_loader.feature_table[self.split] = self.feat_scaler.transform(self.h5_loader.feature_table[self.split])

    def get_labels(self):
        return self.h5_loader.labels[self.split]

    def get_balance(self, balance_type):
        """Return the weight balance for the split of interest.

        Returns: (list) Weights for each label.

        """
        if balance_type == 'balanced_patient':
            labels = self.h5_loader.labels[self.split]
            windows = self.h5_loader.patient_windows[self.split]
            pos_patient = np.array(
                [i for i, k in enumerate(windows) if np.any(labels[k[0]:min(k[0] + self.maxlen, k[1])] == 1)])
            balance = len(pos_patient) / len(windows)
            return [0.5 / (1 - balance), 0.5 / balance]
        elif balance_type == 'balanced_event':
            events = self.h5_loader.events[self.split]
            hazard_labels = np.copy(events)
            hazard_labels[np.where(hazard_labels >= 1)] = 1
            not_first_step = (1 - hazard_labels[1:] + hazard_labels[:-1]) * hazard_labels[
                                                                            1:]  # 1 if previous one is 1 and you are 1
            hazard_labels[np.where(not_first_step)] = np.nan
            _, counts = np.unique(hazard_labels[np.where(~np.isnan(hazard_labels))], return_counts=True)
            return list((1 / counts) * np.sum(counts) / counts.shape[0])
        else:
            labels = self.h5_loader.labels[self.split]
            _, counts = np.unique(labels[np.where(~np.isnan(labels))], return_counts=True)
            return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self):
        """Function to return all the data and labels aligned at once.
        We use this function for the ML methods which don't require a iterator.

        Returns: (np.array, np.array) a tuple containing  data points and label for the split.

        """
        labels = []
        rep = []
        windows = self.h5_loader.patient_windows[self.split][:]
        resampling = self.h5_loader.label_resampling
        logging.info('Gathering the samples for split ' + self.split)
        for start, stop, id_ in tqdm(windows):
            label = self.h5_loader.labels[self.split][start:stop][::resampling][:self.maxlen]
            sample = self.h5_loader.lookup_table[self.split][start:stop][::resampling][:self.maxlen][~np.isnan(label)]
            if self.h5_loader.feature_table is not None:
                features = self.h5_loader.feature_table[self.split][start:stop, 1:][::resampling][:self.maxlen][
                    ~np.isnan(label)]
                sample = np.concatenate((sample, features), axis=-1)
            label = label[~np.isnan(label)]
            if label.shape[0] > 0:
                rep.append(sample)
                labels.append(label)
        rep = np.concatenate(rep, axis=0)
        labels = np.concatenate(labels)
        if self.scaler is not None:
            labels = self.scaler.transform(labels.reshape(-1, 1))[:, 0]
        return rep, labels


@gin.configurable('ICUVariableLengthLoaderTables')
class ICUVariableLengthLoaderTables(object):
    """
    Data loader from h5 compressed files with tables to numpy for variable_size windows.
    """

    def __init__(self, data_path, on_RAM=True, shuffle=True, batch_size=1, splits=('train', 'val'), maxlen=-1, task=0,
                 data_resampling=1, label_resampling=1, use_feat=False, use_presence=False, smooth=False, surv=False,
                 max_horizon=-1):
        """
        Args:
            data_path (string): Path to the h5 data file which should have 3/4 subgroups :data, labels, patient_windows
            and optionally features. Here because arrays have variable length we can't stack them. Instead we
            concatenate them and keep track of the windows in a third file.
            on_RAM (boolean): Boolean whether to load data on RAM. If you don't have ram capacity set it to False.
            shuffle (boolean): Boolean to decide whether to shuffle data between two epochs when using self.iterate
            method. As we wrap this Loader in a torch Dataset this feature is not used.
            batch_size (int): Integer with size of the batch we return. As we wrap this Loader in a torch Dataset this
            is set to 1.
            splits (list): list of splits name . Default is ['train', 'val']
            maxlen (int): Integer with the maximum length of a sequence. If -1 take the maximum length in the data.
            task (int/string/list): Integer with the index of the task we want to train on in the labels.
            If string we find the matching string in data_h5['tasks']. If list, we consider the corresponding
            list of tasks.
            data_resampling (int): Number of step at which we want to resample the data. Default to 1 (5min)
            label_resampling (int): Number of step at which we want to resample the labels (if they exists.
            Default to 1 (5min)
            surv: Boolean to set survival analysis mode where labels correspond to hazard function insteed of eep.
            max_horizon: longest_horizon considered in survival analysis.
        """

        # We set sampling config
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_h5 = tables.open_file(data_path, "r").root
        self.splits = splits
        self.maxlen = maxlen
        self.resampling = data_resampling
        self.label_resampling = label_resampling
        self.use_feat = use_feat
        self.use_presence = use_presence
        self.smooth = smooth
        self.surv = surv
        self.max_horizon = max_horizon
        self.columns = np.array([name.decode('utf-8') for name in self.data_h5['data']['columns'][:]])

        if not isinstance(task, list):
            task = [task]

        # Task given by index(es)
        if isinstance(task[0], int):
            self.task_idx = task
            self.task = None

            if len(self.task_idx) == 1:
                self.task_idx = self.task[0]

        # Task given by string(s)
        else:
            task_ = []
            for t in task:
                task_.append(process_multihorizon_task_string(t))
            self.task = [item for sublist in task_ for item in sublist]
            tasks = list([name.decode('utf-8') for name in self.data_h5['labels']['tasks'][:]])

            if len(self.task) == 1:
                self.task = self.task[0]
                self.task_idx = tasks.index(self.task)
                possible_event = '_'.join(self.task.split('_')[:-1] + ['Event'])

            else:
                self.task_idx = [tasks.index(el) for el in self.task]
                possible_event = '_'.join(self.task[0].split('_')[:-1] + ['Event'])

            if possible_event in tasks:
                self.event_idx = tasks.index(possible_event)
            else:
                self.event_idx = self.task_idx

        self.on_RAM = on_RAM
        # Processing the data part
        if self.data_h5.__contains__('data'):
            if on_RAM:  # Faster but comsumes more RAM
                self.lookup_table = {split: self.data_h5['data'][split][:].astype(np.float32) for split in self.splits}
            else:
                self.lookup_table = {split: self.data_h5['data'][split].astype(np.float32) for split in self.splits}
        else:
            logging.warning('There is no data provided')
            self.lookup_table = None

        # Processing the feature part
        if self.data_h5.__contains__('features') and self.use_feat:
            if on_RAM:  # Faster but comsumes more RAM
                self.feature_table = {split: self.data_h5['features'][split][:] for split in self.splits}
            else:
                self.feature_table = {split: self.data_h5['features'][split] for split in self.splits}
        else:
            self.feature_table = None

        if self.data_h5.__contains__('presence_features') and self.use_presence:
            if on_RAM:  # Faster but comsumes more RAM
                self.presence_table = {split: self.data_h5['presence_features'][split][:] for split in self.splits}
            else:
                self.presence_table = {split: self.data_h5['presence_features'][split] for split in self.splits}
        else:
            self.presence_table = None

        # Processing the label part
        if self.data_h5.__contains__('labels'):
            self.labels = {split: self.data_h5['labels'][split][:, self.task_idx].astype(np.float32) for split in
                           self.splits}
            self.events = {split: self.data_h5['labels'][split][:, self.event_idx].astype(np.float32) for split in
                           self.splits}

            # Some steps might not be labeled so we use valid indexes to avoid them
            self.valid_indexes_labels = {split: np.argwhere(~np.isnan(self.labels[split][:])).T[0]
                                         for split in self.splits}

            self.num_labels = {split: len(self.valid_indexes_labels[split])
                               for split in self.splits}
        else:
            raise Exception('There is no labels provided')

        if self.data_h5.__contains__('patient_windows'):
            # Shape is N_stays x 3. Last dim contains [stay_start, stay_stop, patient_id]
            self.patient_windows = {split: self.data_h5['patient_windows'][split][:].astype(np.int32) for split in
                                    self.splits}
        else:
            raise Exception("patient_windows is necessary to split samples")

        # Iterate counters
        self.current_index_training = {'train': 0, 'test': 0, 'val': 0}

        if self.maxlen == -1:
            seq_lengths = [
                np.max(self.patient_windows[split][:, 1] - self.patient_windows[split][:, 0]) // self.resampling for
                split in
                self.splits]
            self.maxlen = np.max(seq_lengths)
        else:
            self.maxlen = self.maxlen // self.resampling

        # Some patient might have no labeled time points so we don't consider them in valid samples.
        self.valid_indexes_samples = {split: np.array([i for i, k in enumerate(self.patient_windows[split])
                                                       if np.any(
                ~np.isnan(self.labels[split][k[0]:min(k[0] + self.maxlen, k[1])]))])
                                      for split in self.splits}
        self.num_samples = {split: len(self.valid_indexes_samples[split])
                            for split in self.splits}

        if self.smooth:
            self.smooth_labels = {split: self.smooth_labels_split(split) for split in self.splits}

        # if self.surv:
        #     self.surv_labels = {split: self.surv_labels_split(split) for split in self.splits}

        if self.task == 'decomp_24Hours':
            print(self.maxlen)
            print(self.labels[self.splits[0]].dtype)

    def smooth_labels_split(self, split):
        patient_window = self.patient_windows[split]
        labels = self.labels[split]
        events = self.events[split]
        smooth_labels = []
        for start, stop, id_ in tqdm(patient_window):
            label = np.copy(labels[start:stop])
            event = np.copy(events[start:stop])
            not_labeled = np.where(np.isnan(label))
            if len(not_labeled) > 0:
                label[not_labeled] = -1
            if not np.all(label == -1):
                smooth_labels.append(get_smoothed_labels(label, event).astype(np.float32))
            else:
                smooth_labels.append(label.astype(np.float32))
            if len(not_labeled) > 0:
                smooth_labels[-1][not_labeled] = np.nan
        smooth_labels = np.concatenate(smooth_labels, axis=0)
        return smooth_labels

    def surv_labels_split(self, split):
        patient_window = self.patient_windows[split]
        surv_labels = []
        for start, stop, id_ in tqdm(patient_window):
            labels = self.get_hazard_labels(start, stop, split)
            surv_labels.append(labels)
        surv_labels = np.concatenate(surv_labels, axis=0)
        return surv_labels

    def get_window(self, start, stop, split, pad_value=0.0):
        """Windowing function

        Args:
            start (int): Index of the first element.
            stop (int):  Index of the last element.
            split (string): Name of the split to get window from.
            pad_value (float): Value to pad with if stop - start < self.maxlen.

        Returns:
            window (np.array) : Array with data.
            pad_mask (np.array): 1D array with 0 if no labels are provided for the timestep.
            labels (np.array): 1D or 2D array with corresponding labels for each timestep.
            events (np.array): 1D or 2D array with corresponding events labels for each timestep.

        """
        # We resample data frequency
        window = np.copy(self.lookup_table[split][start:stop][::self.resampling])
        labels = np.copy(self.labels[split][start:stop][::self.resampling])
        events = np.copy(self.events[split][start:stop][::self.resampling])

        if self.feature_table is not None:
            feature = np.copy(self.feature_table[split][start:stop][::self.resampling])
            window = np.concatenate([window, feature], axis=-1)
        if self.presence_table is not None:
            presence_feature = np.copy(self.presence_table[split][start:stop][::self.resampling])
            window = np.concatenate([window, presence_feature], axis=-1)
        if self.smooth and split in ['train', 'val']:
            smooth_labels = np.copy(self.smooth_labels[split][start:stop][::self.resampling])
            if len(labels.shape) == 1:  # Single horizon
                labels = np.stack([labels, smooth_labels], axis=-1)
            else:  # Multi horizon
                labels = np.stack([labels, smooth_labels], axis=-1)
        label_resampling_mask = np.zeros((stop - start,))
        label_resampling_mask[::self.label_resampling] = 1.0
        label_resampling_mask = label_resampling_mask[::self.resampling]
        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones((window.shape[0],))

        if length_diff > 0:
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            if len(labels.shape) == 1:
                labels_padding = np.ones((length_diff,)) * pad_value
            else:
                labels_padding = np.ones((length_diff, *labels.shape[1:])) * pad_value
            labels = np.concatenate([labels, labels_padding], axis=0)
            events = np.concatenate([events, np.ones((length_diff,)) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros((length_diff,))], axis=0)
            label_resampling_mask = np.concatenate([label_resampling_mask, np.zeros((length_diff,))], axis=0)

        elif length_diff < 0:
            window = window[:self.maxlen]
            labels = labels[:self.maxlen]
            events = events[:self.maxlen]
            pad_mask = pad_mask[:self.maxlen]
            label_resampling_mask = label_resampling_mask[:self.maxlen]

        not_labeled = np.where(np.isnan(labels))

        # if len(not_labeled) > 0:
        if len(not_labeled[0]) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled[0]] = 0

        # We resample prediction frequency
        pad_mask = pad_mask * label_resampling_mask
        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        events = events.astype(np.float32)
        window = window.astype(np.float32)
        return window, pad_mask, labels, events

    def get_hazard_labels(self, start, stop, split):
        events = np.copy(self.events[split][start:stop][::self.resampling])

        # Handle specific case of Decomp for which event can happen outside of ICU
        if events[-1] > 1:
            events = np.concatenate([events[:-1], np.zeros(int(events[-1]))])
            events[-1] = 1
        early_labels = np.copy(self.labels[split][start:stop][::self.resampling])

        not_first_step = (1 - events[1:] + events[:-1]) * events[1:]  # 1 if previous one is 1 and you are 1
        hazard_labels = np.copy(events[1:])
        hazard_labels[np.where(not_first_step)] = np.nan
        labels = np.copy(np.lib.stride_tricks.sliding_window_view(np.concatenate([hazard_labels,
                                                                                  np.nan * np.ones(self.max_horizon)]),
                                                                  self.max_horizon, writeable=True)[:len(early_labels)])

        # Handle the case where multiple event happened within max_horizon
        pos_idxs = np.where(labels == 1)
        if np.any(labels == 1):
            post_event_idx_1 = np.concatenate(
                [np.arange(i + 1, self.max_horizon) for t, i in zip(pos_idxs[0], pos_idxs[1])]).astype(int)
            post_event_idx_0 = np.concatenate(
                [np.ones(self.max_horizon - i - 1) * t for t, i in zip(pos_idxs[0], pos_idxs[1])]).astype(int)
            labels[post_event_idx_0, post_event_idx_1] = np.nan
        return labels.astype(np.float32)

    def get_window_surv(self, start, stop, split, pad_value=0.0):
        """Windowing function

        Args:
            start (int): Index of the first element.
            stop (int):  Index of the last element.
            split (string): Name of the split to get window from.
            pad_value (float): Value to pad with if stop - start < self.maxlen.

        Returns:
            window (np.array) : Array with data.
            pad_mask (np.array): 1D array with 0 if no labels are provided for the timestep.
            labels (np.array): 1D or 2D array with corresponding HAZARD labels for each timestep.
            early_labels (np.array): 1D or 2D array with corresponding EEP labels for each timestep.
        """
        # We resample data frequency
        window = np.copy(self.lookup_table[split][start:stop][::self.resampling])
        events = np.copy(self.events[split][start:stop][::self.resampling])

        # Handle specific case of Decomp for which event can happen outside of ICU
        if events[-1] > 1:
            events = np.concatenate([events[:-1], np.zeros(int(events[-1]))])
            events[-1] = 1
        early_labels = np.copy(self.labels[split][start:stop][::self.resampling])

        not_first_step = (1 - events[1:] + events[:-1]) * events[1:]  # 1 if previous one is 1 and you are 1
        hazard_labels = np.copy(events[1:])
        hazard_labels[np.where(not_first_step)] = np.nan
        labels = np.copy(np.lib.stride_tricks.sliding_window_view(np.concatenate([hazard_labels,
                                                                                  np.nan * np.ones(self.max_horizon)]),
                                                                  self.max_horizon, writeable=True)[:len(early_labels)])

        # Handle the case where multiple event happened within max_horizon
        pos_idxs = np.where(labels == 1)
        if np.any(labels == 1):
            post_event_idx_1 = np.concatenate(
                [np.arange(i + 1, self.max_horizon) for t, i in zip(pos_idxs[0], pos_idxs[1])]).astype(int)
            post_event_idx_0 = np.concatenate(
                [np.ones(self.max_horizon - i - 1) * t for t, i in zip(pos_idxs[0], pos_idxs[1])]).astype(int)
            labels[post_event_idx_0, post_event_idx_1] = np.nan

        if self.feature_table is not None:
            feature = np.copy(self.feature_table[split][start:stop][::self.resampling])
            window = np.concatenate([window, feature], axis=-1)
        if self.presence_table is not None:
            presence_feature = np.copy(self.presence_table[split][start:stop][::self.resampling])
            window = np.concatenate([window, presence_feature], axis=-1)
        label_resampling_mask = np.zeros((stop - start,))
        label_resampling_mask[::self.label_resampling] = 1.0
        label_resampling_mask = label_resampling_mask[::self.resampling]
        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones_like(labels)

        if length_diff > 0:
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels_padding = np.ones((length_diff, *labels.shape[1:])) * -1  # we pad labels with -1 not pad_value
            labels = np.concatenate([labels, labels_padding], axis=0)
            e_labels_padding = np.ones(
                (length_diff, *early_labels.shape[1:])) * -1  # we pad labels with -1 not pad_value
            early_labels = np.concatenate([early_labels, e_labels_padding], axis=0)
            events = np.concatenate([events, np.ones((length_diff,)) * -1], axis=0)  # same for events
            pad_mask = np.concatenate([pad_mask, np.zeros((length_diff, *labels.shape[1:]))], axis=0)
            label_resampling_mask = np.concatenate([label_resampling_mask, np.zeros((length_diff,))], axis=0)

        elif length_diff < 0:
            window = window[:self.maxlen]
            labels = labels[:self.maxlen]
            events = events[:self.maxlen]
            pad_mask = pad_mask[:self.maxlen]
            early_labels = early_labels[:self.maxlen]
            label_resampling_mask = label_resampling_mask[:self.maxlen]

        not_labeled = np.where(np.isnan(labels))
        not_pred = np.where(np.isnan(early_labels))

        if len(not_labeled[0]) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        if len(not_pred[0]) > 0:
            labels[not_pred] = -1
            pad_mask[not_pred] = 0

        # We resample prediction frequency
        pad_mask = pad_mask * label_resampling_mask.reshape(-1, 1)
        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        events = events.astype(np.float32)
        window = window.astype(np.float32)
        if len(np.where((pad_mask == 1) * (labels == -1))[0]) > 0:
            raise Exception("Mismatch between mask and labeling")
        return window, pad_mask, labels, early_labels

    def sample(self, random_state, split='train', idx_patient=None):
        """Function to sample from the data split of choice.
        Args:
            random_state (np.random.RandomState): np.random.RandomState instance for the idx choice if idx_patient
            is None.
            split (string): String representing split to sample from, either 'train', 'val' or 'test'.
            idx_patient (int): (Optional) Possibility to sample a particular sample given a index.
        Returns:
            A sample from the desired distribution as tuple of numpy arrays (sample, label, mask).
        """

        assert split in self.splits

        if idx_patient is None:
            idx_patient = random_state.randint(self.num_samples[split], size=(self.batch_size,))
            state_idx = self.valid_indexes_samples[split][idx_patient]
        else:
            state_idx = self.valid_indexes_samples[split][idx_patient]

        patient_windows = self.patient_windows[split][state_idx]

        X = []
        y = []
        pad_masks = []
        e = []
        if self.batch_size == 1:
            if self.surv:
                X, pad_masks, y, e = self.get_window_surv(patient_windows[0], patient_windows[1], split)
            else:
                X, pad_masks, y, e = self.get_window(patient_windows[0], patient_windows[1], split)
            return X, pad_masks, y, e
        else:
            for start, stop, id_ in patient_windows:
                window, pad_mask, labels, events = self.get_window(start, stop, split)
                X.append(window)
                y.append(labels)
                e.append(events)
                pad_masks.append(pad_mask)
            X = np.stack(X, axis=0)
            pad_masks = np.stack(pad_masks, axis=0)
            y = np.stack(y, axis=0)
            e = np.stack(e, axis=0)
            return X, pad_masks, y, e
