import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold


from data.transforms import get_transfos
from data.dataset import PatientDataset
from model_zoo.models import define_model
from utils.torch_utils import load_model_weights
from params import NUM_WORKERS, DATA_PATH, FEATURES_PATH, IMG_PATH


def extract_features(model, dataset, batch_size=4):
    model.eval()
    fts = np.empty((0, model.nb_ft))
    preds = np.empty(0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    with torch.no_grad():
        for x, _ in loader:
            y, ft = model.extract_ft(x.cuda())
            fts = np.concatenate([fts, ft.detach().cpu().numpy()])
            preds = np.concatenate([preds, torch.sigmoid(y).detach().cpu().numpy()])

    return preds, fts


if __name__ == "__main__":
    # Load data

    df = pd.read_csv(DATA_PATH + "train.csv")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X=df, y=df, groups=df["StudyInstanceUID"]))

    fold_idx = np.zeros(len(df))
    for i, (train_idx, val_idx) in enumerate(splits):
        fold_idx[val_idx] = i
    df["fold"] = fold_idx

    paths = [
        IMG_PATH + f"{study}/{series}/"
        for study, series in df[["StudyInstanceUID", "SeriesInstanceUID"]].values
    ]
    df["path"] = paths
    unique_df = df[
        ["path", "StudyInstanceUID", "SeriesInstanceUID", "fold"]
    ].drop_duplicates()

    # Load models

    CP_PATH = "../weights/"
    weights = [f for f in sorted(os.listdir(CP_PATH)) if "efficientnet" in f]

    models = []

    for weight in weights:
        model = define_model("efficientnet-b3").cuda()
        model = load_model_weights(model, CP_PATH + weight)
        models.append(model)

    # Extract features

    SAVE_PATH = FEATURES_PATH + "b3/"

    transforms = get_transfos(augment=False)

    for path, study, series, fold in tqdm(unique_df.values):
        dataset = PatientDataset(path, transforms)

        preds, features = extract_features(models[int(fold)], dataset, batch_size=32)

        np.save(
            f"{SAVE_PATH}/features_{'_'.join(path.split('/')[-3:-1])}.npy", features
        )
        np.save(f"{SAVE_PATH}/preds_{'_'.join(path.split('/')[-3:-1])}.npy", preds)

        # break
