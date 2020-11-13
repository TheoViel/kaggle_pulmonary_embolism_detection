import os
import cv2
import torch
import numpy as np

from tqdm.notebook import tqdm
from torch.utils.data import Dataset

from params import IMG_PATH, FEATURES_PATH, IMG_TARGET, EXAM_TARGETS, MAX_LEN


def sop_to_img(sop, folder):
    files = os.listdir(folder)
    for img in files:
        if img.endswith(sop + ".jpg"):
            return img

    print("Not found")
    return np.random.choice(files)


def get_img_names(df):
    img_names = []
    studies = df["StudyInstanceUID"].values
    series = df["SeriesInstanceUID"].values
    sops = df["SOPInstanceUID"].values

    for idx in tqdm(range(len(df))):
        folder = IMG_PATH + "/" + studies[idx] + "/" + series[idx] + "/"
        img_name = sop_to_img(sops[idx], folder)

        img_names.append(folder + img_name)

    return img_names


class PEDatasetImg(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

        self.patients = [
            p + "_" + z for p, z in df[["StudyInstanceUID", "SeriesInstanceUID"]].values
        ]

        self.img_paths = df["img_path"].values
        self.targets = self.df[IMG_TARGET].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        y = torch.tensor(self.targets[idx], dtype=torch.float)

        return image, y


class PatientDataset(Dataset):
    """
    Dataset for feature extraction
    """

    def __init__(self, path, transforms=None):
        self.path = path
        self.img_paths = sorted(os.listdir(path))

        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.path + self.img_paths[idx])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, idx


class PEDatasetFt(Dataset):
    def __init__(self, df, paths=[FEATURES_PATH], max_len=MAX_LEN):
        self.df = df
        self.paths = [[path + p for path in paths] for p in self.df["path"].values]

        self.max_len = max_len

        self.img_targets = df[IMG_TARGET].values
        self.exam_targets = df[EXAM_TARGETS].values

        self.img_targets = []
        for t in df[IMG_TARGET].values:
            self.img_targets.append(self.pad(np.array(t)))
        self.img_targets = np.array(self.img_targets)

    def __len__(self):
        return len(self.df)

    def pad(self, x):
        length = x.shape[0]
        if length > self.max_len:
            return x[: self.max_len]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[:length] = x
            return padded

    def __getitem__(self, idx):
        ft = np.concatenate([np.load(p) for p in self.paths[idx]], -1)
        size = min(ft.shape[0], self.max_len)
        ft = self.pad(ft)

        return (
            torch.tensor(ft, dtype=torch.float),
            torch.tensor(self.exam_targets[idx], dtype=torch.float),
            torch.tensor(self.img_targets[idx], dtype=torch.float),
            torch.tensor(size, dtype=torch.int64),
        )
