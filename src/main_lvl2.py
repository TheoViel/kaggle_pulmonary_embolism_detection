import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

from data.dataset import PEDatasetFt
from utils.metric import rsna_metric
from training.train_lvl2 import k_fold
from utils.logger import create_logger, prepare_log_folder
from params import FEATURES_PATH, IMG_TARGET, DATA_PATH, EXAM_TARGETS, IMG_PATH, LOG_PATH_2


def str_to_arr(x):
    x = re.sub("\n", " ", x[1:-1])
    x = re.sub(r"\.", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return np.array(x.split(" ")).astype(int)


class Config:
    """
    Parameters used for training
    """

    # General
    seed = 42
    verbose = 1
    save_weights = True
    max_len = 400

    ft_path = [
        FEATURES_PATH + "b3/",
    ]

    # k-fold
    k = 5
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    ft_dim = 1536
    lstm_dim = 256
    dense_dim = 256
    logit_dim = 256
    use_msd = True

    # Training
    loss = "BCEWithLogitsLoss"
    optimizer = "Adam"

    batch_size = 32
    epochs = 10
    swa_first_epoch = 7
    lr = 5e-3
    warmup_prop = 0.0
    val_bs = 32

    name = "rnn"


if __name__ == "__main__":
    # Loading data

    try:  # read already computed data if possible.
        df = pd.read_csv("../output/df_patient_level.csv")
        df[IMG_TARGET] = df[IMG_TARGET].apply(str_to_arr)

    except FileNotFoundError:
        df = pd.read_csv(DATA_PATH + "train.csv")
        df = (
            df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])[
                ["SOPInstanceUID"] + EXAM_TARGETS + [IMG_TARGET]
            ]
            .agg(list)
            .reset_index()
        )

        ordered_targets = []
        for study, series, names, tgt in tqdm(
            df[
                [
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "SOPInstanceUID",
                    "pe_present_on_image",
                ]
            ].values
        ):
            imgs = sorted(os.listdir(IMG_PATH + f"{study}/{series}/"))
            ordered_names = [n.split("_")[1][:-4] for n in imgs]
            ordered_target = np.zeros(len(ordered_names))

            for name, t in zip(names, tgt):
                ordered_target[ordered_names.index(name)] = t

            ordered_targets.append(ordered_target)
        df[IMG_TARGET] = ordered_targets

        for c in EXAM_TARGETS:
            df[c] = df[c].apply(lambda x: x[0])

        df.to_csv("../output/df_patient_level.csv", index=False)

    df["path"] = (
        "features_" + df["StudyInstanceUID"] + "_" + df["SeriesInstanceUID"] + ".npy"
    )
    df["path_preds"] = (
        "preds_" + df["StudyInstanceUID"] + "_" + df["SeriesInstanceUID"] + ".npy"
    )

    # Logging

    log_folder = prepare_log_folder(LOG_PATH_2)
    print(f"Logging results to {log_folder}")

    create_logger(directory=log_folder, name="logs.txt")

    pred_exams_oof, pred_imgs_oof, sizes_oof = k_fold(Config, df, log_folder)

    dataset = PEDatasetFt(df, [FEATURES_PATH + "b3/"])
    cv = rsna_metric(
        dataset.img_targets,
        dataset.exam_targets,
        pred_imgs_oof,
        pred_exams_oof,
        sizes_oof,
    )

    print(f"Local CV score : {cv:.4f}")
