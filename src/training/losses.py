import torch
import torch.nn as nn

LOSSES = ["CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"]


def define_loss(name):
    if name in LOSSES:
        loss = getattr(torch.nn, name)(reduction="mean")
    else:
        raise NotImplementedError

    return loss


def prepare_for_loss(y_pred, y_batch, loss):
    if loss == "BCEWithLogitsLoss":
        y_pred = y_pred.view(-1)
    else:
        raise NotImplementedError
    return y_pred, y_batch.cuda()


class RSNAWLoss(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.device = torch.device("cuda") if cuda else torch.device("cpu")

        self.label_w = (
            torch.tensor(
                [
                    0.0736196319,
                    0.2346625767,
                    0.0782208589,
                    0.06257668712,
                    0.1042944785,
                    0.06257668712,
                    0.1042944785,
                    0.1877300613,
                    0.09202453988,
                ]
            )
            .view(1, -1)
            .to(self.device)
        )

        self.img_w = 0.07361963
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes):
        total_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        total_weights = torch.tensor(0, dtype=torch.float32).to(self.device)

        for y_img, y_exam, pred_img, pred_exam, size in zip(
            y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes
        ):

            y_img = y_img[: len(pred_img)]

            exam_loss = self.bce(pred_exam, y_exam)
            exam_loss = torch.sum(exam_loss * self.label_w, 1)[0]

            image_loss = self.bce(pred_img, y_img).mean()
            qi = torch.sum(y_img)
            image_loss = torch.sum(self.img_w * qi * image_loss)

            total_loss += exam_loss + image_loss
            total_weights += self.label_w.sum() + self.img_w * qi

        final_loss = total_loss / total_weights
        return final_loss
