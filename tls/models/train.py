import os
import random
import shutil
import sys

import torch

from tls.data.loader import *
from tls.models.utils import save_config_file


def train_with_gin(model_dir=None,
                   overwrite=False,
                   load_weights=False,
                   gin_config_files=None,
                   gin_bindings=None,
                   seed=1234,
                   reproducible=True):
    """Trains a model based on the provided gin configuration.
    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.
    Args:
        model_dir: String with path to directory where model output should be saved.
        overwrite: Boolean indicating whether to overwrite output directory.
        gin_config_files: List of gin config files to load.
        gin_bindings: List of gin bindings to use.
        seed: Integer corresponding to the common seed used for any random operation.
    """

    # Setting the seed before gin parsing
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reproducible results on the same hardware
    if reproducible:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True) # Had to set that to false because of torch.gather
        torch.backends.cudnn.benchmark = False

    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    train_common(model_dir, overwrite, load_weights)
    gin.clear_config()


@gin.configurable('train_common')
def train_common(log_dir, overwrite=False, load_weights=False, model=gin.REQUIRED, dataset_fn=gin.REQUIRED,
                 data_path=gin.REQUIRED, weight=None, do_test=False):
    """
    Common wrapper to train all benchmarked models.
    """
    if os.path.isdir(log_dir) and not load_weights:
        if overwrite or (not os.path.isfile(os.path.join(log_dir, 'test_metrics.pkl'))):
            shutil.rmtree(log_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    if not load_weights:
        os.makedirs(log_dir)
    dataset = dataset_fn(data_path, split='train')
    val_dataset = dataset_fn(data_path, split='val')

    # We set the label scaler
    val_dataset.set_scaler(dataset.scaler)
    model.set_scaler(dataset.scaler)

    if dataset.scale_feature:
        val_dataset.set_feat_scaler(dataset.feat_scaler)

    model.set_logdir(log_dir)
    save_config_file(log_dir)  # We save the operative config before and also after training

    if load_weights:
        if os.path.isfile(os.path.join(log_dir, 'model.torch')):
            model.load_weights(os.path.join(log_dir, 'model.torch'))
        else:
            raise Exception("No weights to load at path : {}".format(os.path.join(log_dir, 'model.torch')))
        do_test = True

    else:
        try:
            model.train(dataset, val_dataset, weight)
        except ValueError as e:
            logging.exception(e)
            if 'Only one class present' in str(e):
                logging.error("There seems to be a problem with the evaluation metric. In case you are attempting "
                              "to train with the synthetic data, this is expected behaviour")
            sys.exit(1)

    del dataset.h5_loader.lookup_table
    del val_dataset.h5_loader.lookup_table

    if do_test:
        test_dataset = dataset_fn(data_path, split='test')
        test_dataset.set_scaler(dataset.scaler)
        if weight is not None:
            weight = dataset.get_balance('class')
        if dataset.scale_feature:
            test_dataset.set_feat_scaler(dataset.feat_scaler)
        model.test(test_dataset, weight)
        del test_dataset.h5_loader.lookup_table
    save_config_file(log_dir)
