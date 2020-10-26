import gc
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm.notebook import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import get_linear_schedule_with_warmup

from params import NUM_WORKERS
from training.sampler import PatientSampler
from training.optimizer import define_optimizer
from training.losses import define_loss, prepare_for_loss


def fit(
    model,
    train_dataset,
    val_dataset,
    samples_per_patient=10, 
    optimizer_name='adam',
    loss_name='bce',
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
        samples_per_patient=samples_per_patient
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=NUM_WORKERS) 

    print(f"Using {len(train_loader)} out of {len(train_dataset) // batch_size} batches by limiting to {samples_per_patient} samples per patient.\n")

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
        avg_val_loss = 0.
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
                f"Epoch {epoch + 1:02d}/{epochs} \t lr={lr:.1e} \t t={elapsed_time:.0f}s  \t loss={avg_loss:.3f} \t ",
                end="",
            )
            print(
                f"val_loss={avg_val_loss:.3f}"
            )

    torch.cuda.empty_cache()


def predict(model, dataset, activation="sigmoid", batch_size=64, num_classes=1):
    """
    Usual torch predict function
    Arguments:
        model {torch model} -- Model to predict with
        dataset {torch dataset} -- Dataset to predict with on
    Keyword Arguments:
        batch_size {int} -- Batch size (default: {32})
    Returns:
        numpy array -- Predictions
    """
    model.eval()
    preds = np.empty((0))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    with torch.no_grad():
        for x, _ in loader:
            y_pred = model(x.cuda()).detach()

            if activation == "sigmoid":
                y_pred = torch.sigmoid(y_pred)
            elif activation == "softmax":
                y_pred = torch.softmax(y_pred, -1)

            preds = np.concatenate([preds, y_pred.cpu().numpy()])

    return preds
