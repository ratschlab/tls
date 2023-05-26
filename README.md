# Temporal Label Smoothing

This repository contains the code for the paper:
**Temporal Label Smoothing for Early Prediction of Adverse Events**. It contains code from both datasets' original
repositories, [M3B](https://github.com/YerevaNN/mimic3-benchmarks)
and [HiB](https://github.com/ratschlab/HIRID-ICU-Benchmark), which we extended to extract labels at multiple horizons or
add additional components.

## Set up the environment

For all our experiments, we assume a Linux installation, however, other platforms may also work:

1. Install Conda, see
   the [official installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. clone this repository and change it into the directory of the repository
3. `conda env update` (creates an environment `tls-env` using the `environment.yml` file)
4. `pip install -e .` (creates package `tls`)

## Preprocess the data

### Download the raw data

#### HiRID

1. Get access to the HiRID 1.1.1 dataset on [physionet](https://physionet.org/content/hirid/1.1.1/). This entails
    1. getting a [credentialed physionet account](https://physionet.org/settings/credentialing/)
    2. [submit a usage request](https://physionet.org/request-access/hirid/1.1.1/) to the data depositor
2. Once access is granted, download the following files
    1. [reference_data.tar.gz](https://physionet.org/content/hirid/1.1.1/reference_data.tar.gz)
    2. [observation_tables_parquet.tar.gz](https://physionet.org/content/hirid/1.1.1/raw_stage/observation_tables_parquet.tar.gz)
    3. [pharma_records_parquet.tar.gz](https://physionet.org/content/hirid/1.1.1/raw_stage/pharma_records_parquet.tar.gz)
3. unpack the files into the `hirid-data-root` directory using e.g. `cat *.tar.gz | tar zxvf - -i`

#### MIMIC-III

1. Get access to MIMIC-III dataset on [physionet](https://physionet.org/content/mimiciii/1.4/)
    1. getting a [credentialed physionet account](https://physionet.org/settings/credentialing/)
    2. complete required [training](https://physionet.org/settings/training/)
    3. sign the data use [agreement](https://physionet.org/sign-dua/mimiciii/1.4/)
2. Once access is granted, download all `CSV` files provided on the page and place them in a directory `mimic3-source`
3. Run all the steps described in [M3B repository](https://github.com/YerevaNN/mimic3-benchmarks) to obtain *MIMIC-III
   Benchmark* data. You should place this data in the so-called `mimic3-data-root` folder.

### Run the data pipeline

Here we describe how to obtain the dataset in a format compatible with the deep learning models we use.

#### HiB

You can directly obtain our preprocessed version of the HiB dataset with the following steps:

1. Activate the conda environment using `conda activate tls-env`.
2. Complete the arguments in `run_script/preprocess/hirid.sh` for `--hirid-data-root` and `--work-dir`.
3. Run pre-processing with `sh run_script/preprocess/hirid.sh`

This second step wraps the following command that you can adapt to your need.

```
tls preprocess --dataset hirid \
               --hirid-data-root [path to source] #TODO User \
               --work-dir [path to output] #TODO User \
               --resource-path ./preprocessing/resources/ \
               --horizons 2 4 6 8 10 12 14 16 18 20 22 \
	       --nr-worker 8
```

The above command requires about 10GB of RAM per core and, in total, approximately 40GB of disk space.

#### M3B

Similarly, you can directly obtain our preprocessed version of the M3B dataset with the following steps:

1. Activate the conda environment using `conda activate tls-env`.
2. Complete the arguments in `run_script/preprocess/mimic3.sh` for `--mimic3-data-root` and `--work-dir`.
3. Run pre-processing with `sh run_script/preprocess/mimic3.sh`

This second step wraps the following command that you can adapt to your need.

```
tls preprocess --dataset mimic3 \
               --mimic3-data-root [path to source] #TODO User \
               --work-dir [path to output] #TODO User \
               --resource-path ./preprocessing/resources/ \
               --horizons 4 8 12 16 20 24 28 32 36 40 44 \
               --mimic3-static-columns Height
```

The above command requires about 10GB of RAM per core and, in total, approximately 20GB of disk space.

## Run Experiments
### Update config files
The code is built around [gin-config]() files. These files needs to be modified with the source path to the data. 
You should update the files in `./configs` where you there is a `#TODO User` as in the previous step. 
For instance in `./configs/hirid/GRU.gin` you should insert the correct path at line 36:
```
train_common.data_path = [path to output of pipe] #TODO User
```

### Reproduce experiments from the paper

If you are interested in reproducing the experiments from the paper, you can directly use the pre-built scripts
in `./run_scripts/`. For instance, you can run the following command to reproduce the GRU baseline on the Circulatory
Failure task:

```
sh run_script/baseline/Circ/GRU.sh
```

this will create a new directory `[path to logdir]/[task name]/[seed number]/` containing:

- `val_metrics.pkl` and `test_metrics.pkl`: Pickle files with the model's performance on respective validation and test
  sets.
- `train_config.gin`: The so-called "operative" config allows the saving of the configuration used at training.
- `model.torch` : The weights of the model that was trained.
- `tensorboard/`: (Optional) Directory with tensorboard logs. One can do `tensorboard --logdir ./tensorboard` to
  visualize them.

The pre-built scripts are divided into two categories as follows:

- `baseline`: This folder contains scripts to reproduce the main benchmark experiment. Each of them will run a model
  with the best parameters we provide for ten identical seeds.
- `hp-search`:  This folder contains the scripts we used to search hyperparameters for our method and baselines.

### Run evaluation of trained models

For a trained model, you can evaluate any previously trained model using the `evaluate` as follows:

```
tls evaluate -c [path to gin config] \
             -l [path to logdir] \
             -t [task name] \
```

This command will evaluate the model at `[path to logdir]/[task name]/model.torch` on the test set of the dataset
provided in the config. Results are saved to the `test_metrics.pkl` file. 

