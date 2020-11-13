import torch
import numpy as np

from time import time
from torchcontrib.optim import SWA
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from transformers import get_linear_schedule_with_warmup

from params import NUM_EXAM_TARGETS
from utils.metric import rsna_metric
from data.dataset import PEDatasetFt
from training.losses import RSNAWLoss
from model_zoo.models_lvl2 import RNNModel
from training.optimizer import define_optimizer
from utils.torch_utils import save_model_weights, seed_everything, count_parameters


def fit(
    model,
    train_dataset,
    val_dataset,
    optimizer_name="adam",
    loss_name="bce",
    epochs=10,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    swa_first_epoch=10,
    verbose=1,
):

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)
    if swa_first_epoch < epochs:
        optimizer = SWA(optimizer)

    loss_fct = RSNAWLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        start_time = time()

        avg_loss = 0

        for x, y_exam, y_img, sizes in train_loader:
            pred_exam, pred_img = model(x.cuda())

            loss = loss_fct(
                y_img.cuda(), y_exam.cuda(), pred_img, pred_exam, sizes.cuda()
            )
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        if epoch + 1 >= swa_first_epoch:
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        model.eval()
        avg_val_loss = 0.0
        sizes = np.empty((0))
        pred_exams = np.empty((0, NUM_EXAM_TARGETS))
        pred_imgs = np.empty((0, val_dataset.max_len))

        with torch.no_grad():
            for x, y_exam, y_img, size in val_loader:
                pred_exam, pred_img = model(x.cuda())

                loss = loss_fct(
                    y_img.cuda(),
                    y_exam.cuda(),
                    pred_img.detach(),
                    pred_exam.detach(),
                    size.cuda(),
                )

                avg_val_loss += loss.item() / len(val_loader)

                pred_exams = np.concatenate(
                    [pred_exams, torch.sigmoid(pred_exam).detach().cpu().numpy()]
                )
                pred_imgs = np.concatenate(
                    [pred_imgs, torch.sigmoid(pred_img).detach().cpu().numpy()]
                )
                sizes = np.concatenate([sizes, size.numpy()])

        score = rsna_metric(
            val_dataset.img_targets,
            val_dataset.exam_targets,
            pred_imgs,
            pred_exams,
            sizes,
        )

        if epoch + 1 >= swa_first_epoch and epoch < epochs - 1:
            optimizer.swap_swa_sgd()

        elapsed_time = time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e} \t t={elapsed_time:.0f}s"
                f"\t loss={avg_loss:.3f} \t ",
                end="",
            )
            print(f"val_loss={avg_val_loss:.3f}\t score={score:.4f}")

    torch.cuda.empty_cache()

    return pred_exams, pred_imgs, sizes


def train(config, df_train, df_val, fold, log_folder=""):
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

    model = RNNModel(
        ft_dim=config.ft_dim,
        lstm_dim=config.lstm_dim,
        dense_dim=config.dense_dim,
        logit_dim=config.logit_dim,
        use_msd=config.use_msd,
    ).cuda()

    model.zero_grad()

    train_dataset = PEDatasetFt(df_train, max_len=config.max_len, paths=config.ft_path)
    val_dataset = PEDatasetFt(df_val, max_len=config.max_len, paths=config.ft_path)

    n_parameters = count_parameters(model)
    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_exams, pred_imgs, sizes = fit(
        model,
        train_dataset,
        val_dataset,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        swa_first_epoch=config.swa_first_epoch,
    )

    if config.save_weights:
        save_model_weights(
            model,
            f"{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    return pred_exams, pred_imgs, sizes


def k_fold(config, df, log_folder=""):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder :
    oof predictions, val predictions, val indices, histories

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        log_folder (str, optional): Folder to logs results to. Defaults to ''.
    """

    pred_exams_oof = np.zeros((len(df), NUM_EXAM_TARGETS))
    pred_imgs_oof = np.zeros((len(df), config.max_len))
    sizes_oof = np.zeros(len(df))

    kf = KFold(n_splits=config.k)
    splits = list(kf.split(X=df))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            pred_exams, pred_imgs, sizes = train(
                config, df_train, df_val, i, log_folder=log_folder
            )

            pred_exams_oof[val_idx] = pred_exams
            pred_imgs_oof[val_idx] = pred_imgs
            sizes_oof[val_idx] = sizes

    return pred_exams_oof, pred_imgs_oof, sizes_oof
