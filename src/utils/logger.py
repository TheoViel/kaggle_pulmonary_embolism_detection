import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

from params import LOG_PATH

LOGGED_IN_CONFIG = []


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file
    Keyword Arguments:
        directory {str} -- Path to save logs at (default: {''})
        name {str} -- Name of the file to save the logs in (default: {'logs.txt'})
    """
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exp_id = len(os.listdir(log_today))
    log_folder = log_today + f"{exp_id}/"

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    else:
        print("Experiment already exists")

    return log_folder


def save_config(config, path):
    dic = config.__dict__.copy()
    del dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"]

    with open(path + ".json", "w") as f:
        json.dump(dic, f)

    dic["selected_folds"] = [", ".join(np.array(dic["selected_folds"]).astype(str))]
    config_df = pd.DataFrame.from_dict(dic)
    config_df.to_csv(path + ".csv", index=False)

    return config_df

