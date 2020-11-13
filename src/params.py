import os
import torch
import warnings
import numpy as np

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 2020

DATA_PATH = "../input/"
IMG_PATH = "../../../data/pe/train-jpegs/"
FEATURES_PATH = "../../../data/pe/train-fts/"

LOG_PATH = "../logs/"
LOG_PATH_2 = "../logs2/"


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

SIZE = 256
MAX_LEN = 1083

IMG_TARGET = "pe_present_on_image"

EXAM_TARGETS = [
    "negative_exam_for_pe",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "leftsided_pe",
    "chronic_pe",
    "rightsided_pe",
    "acute_and_chronic_pe",
    "central_pe",
    "indeterminate",
]

NUM_EXAM_TARGETS = len(EXAM_TARGETS)
