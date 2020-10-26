# From https://www.kaggle.com/khyeh0719/0929-updated-rsna-competition-metric

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


CFG = {
    "image_weight": 0.07361963,
    "exam_weights": [
        0.0736196319,
        0.2346625767,
        0.0782208589,
        0.06257668712,
        0.1042944785,
        0.06257668712,
        0.1042944785,
        0.1877300613,
        0.09202453988,
    ],
}


def bce(pred, truth):
    return -(truth * np.log(pred) + (1.0 - truth) * np.log(1.0 - pred))


def rsna_metric(y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes):
    label_w = np.array(CFG["exam_weights"]).reshape((1, -1))
    img_w = CFG["image_weight"]
    total_loss = 0.
    total_weights = 0.

    for y_img, y_exam, pred_img, pred_exam, size in zip(
                    y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes
                ):
        
        y_img = y_img[:len(pred_img)]
        
        exam_loss = bce(pred_exam, y_exam)
        exam_loss = np.sum(exam_loss * label_w, 1)[0]
        
        qi = np.sum(y_img)
        image_loss = bce(pred_img, y_img).mean()
        image_loss = np.sum(img_w * qi * image_loss)
        
        total_loss += exam_loss + image_loss
        total_weights += label_w.sum() + img_w * qi

    final_loss = total_loss / total_weights
    return final_loss
