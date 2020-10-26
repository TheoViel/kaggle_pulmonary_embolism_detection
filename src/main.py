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


def train(config, df_train, df_val, fold, log_folder=''):
    """
    Trains and validate a model

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (str, optional): Folder to logs results to. Defaults to ''.

    Returns:
        np array: Validation predictions.
        pandas dataframe: Training history.
    """

    seed_everything(config.seed)

    model = define_model(
        config.selected_model, 
    ).cuda()
    model.zero_grad()

    train_dataset = PEDatasetImg(df_train, transforms=get_transfos())
    val_dataset = PEDatasetImg(df_val, transforms=get_transfos(augment=False))
        
    n_parameters = count_parameters(model)
    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    fit(
        model,
        train_dataset,
        val_dataset,
        samples_per_patient=config.samples_per_patient,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
    )

    if config.save_weights:
        save_model_weights(
            model,
            f"{config.selected_model}_{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )


def k_fold(config, df, log_folder=''):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder :
    oof predictions, val predictions, val indices, histories

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        log_folder (str, optional): Folder to logs results to. Defaults to ''.
    """

    gkf = GroupKFold(n_splits=config.k)
    splits = list(gkf.split(X=df, y=df, groups=df['StudyInstanceUID']))


    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy().sample(10000)  # 10k samples is enough to evaluate

            train(config, df_train, df_val, i, log_folder=log_folder)
            
    # TODO : feature extraction


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
