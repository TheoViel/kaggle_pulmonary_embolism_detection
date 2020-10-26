import os
import datetime
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import GroupKFold


from params import *
from utils.logger import *
from data.dataset import *
from training.train import *
from utils.torch_utils import *
from data.transforms import get_transfos
from model_zoo.models import define_model


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    save_weights = True

    # k-fold
    k = 5
    selected_folds = [4]

    # Model
    # selected_model = "resnext50_32x4d"
    selected_model = "efficientnet-b3"

    # Training
    samples_per_patient = 30
    loss = "BCEWithLogitsLoss"
    optimizer = "Adam"
    
    batch_size = 16 if '101' in selected_model else 32
    epochs = 15
    lr = 5e-4 if '101' in selected_model else 1e-3
    warmup_prop = 0.05
    val_bs = 32

    name = ""


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH + "train.csv")

    log_folder = prepare_log_folder(LOG_PATH)
    print(f'Logging results to {log_folder}')

    config_df = save_config(Config, log_folder + 'config.json')

    create_logger(directory=log_folder, name="logs.txt")

    k_fold(Config, df, log_folder)
