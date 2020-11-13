import time
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from torch.utils.data.sampler import RandomSampler
from transformers import get_linear_schedule_with_warmup

from params import NUM_WORKERS

from data.dataset import PEDatasetImg
from data.transforms import get_transfos
from model_zoo.models import define_model
from training.sampler import PatientSampler
from training.optimizer import define_optimizer
from training.losses import define_loss, prepare_for_loss

from utils.torch_utils import seed_everything, save_model_weights, count_parameters


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

    gkf = GroupKFold(n_splits=config.k)
    splits = list(gkf.split(X=df, y=df, groups=df["StudyInstanceUID"]))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = (
                df.iloc[val_idx].copy().sample(10000)
            )  # 10k samples is enough to evaluate

            train(config, df_train, df_val, i, log_folder=log_folder)


def fit(
    model,
    train_dataset,
    val_dataset,
    samples_per_patient=10,
    optimizer_name="adam",
    loss_name="bce",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    verbose=1,
):

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)

    loss_fct = define_loss(loss_name).cuda()

    sampler = PatientSampler(
        RandomSampler(train_dataset),
        train_dataset.patients,
        batch_size=batch_size,
        drop_last=True,
        samples_per_patient=samples_per_patient,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=NUM_WORKERS
    )

    print(
        f"Using {len(train_loader)} out of {len(train_dataset) // batch_size} "
        f"batches by limiting to {samples_per_patient} samples per patient.\n"
    )

    val_loader = DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False, num_workers=NUM_WORKERS
    )

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        start_time = time.time()

        avg_loss = 0
        for x, y_batch in train_loader:

            y_pred = model(x.cuda()).float()

            y_pred, y_batch = prepare_for_loss(y_pred, y_batch, loss_name)
            loss = loss_fct(y_pred, y_batch)

            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            model.zero_grad()
            scheduler.step()

        model.eval()
        avg_val_loss = 0.0
        with torch.no_grad():
            for x, y_batch in val_loader:
                y_pred = model(x.cuda()).detach()

                y_pred, y_batch = prepare_for_loss(y_pred, y_batch, loss_name)
                loss = loss_fct(y_pred, y_batch)
                avg_val_loss += loss.item() / len(val_loader)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs} \t lr={lr:.1e} \t"
                f"t={elapsed_time:.0f}s  \t loss={avg_loss:.3f} \t ",
                end="",
            )
            print(f"val_loss={avg_val_loss:.3f}")

    torch.cuda.empty_cache()
