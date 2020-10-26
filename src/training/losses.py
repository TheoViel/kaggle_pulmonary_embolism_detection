import torch
import torch.nn as nn 

LOSSES = ["L1Loss", "MSELoss", "CrossEntropyLoss", "CTCLoss", "NLLLoss", "PoissonNLLLoss", "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "MarginRankingLoss", "HingeEmbeddingLoss", "MultiLabelMarginLoss", "SmoothL1Loss", "SoftMarginLoss", "MultiLabelSoftMarginLoss", "CosineEmbeddingLoss", "MultiMarginLoss", "TripletMarginLoss"]

def define_loss(name):
    try:
        loss = getattr(torch.nn, name)(reduction='mean')
    except:
        raise NotImplementedError

    return loss


def prepare_for_loss(y_pred, y_batch, loss):
    if loss == "BCEWithLogitsLoss":
        y_pred = y_pred.view(-1)
    else:
        pass
    return y_pred, y_batch.cuda()


# class RSNAWLoss(nn.Module):
#     def __init__(self, cuda=True):
#         super().__init__()
#         self.device = torch.device("cuda") if cuda else torch.device("cpu")

#         self.label_w = torch.tensor([
#             0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 
#             0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988
#         ]).view(1, -1).to(self.device)

#         self.img_w = 0.07361963
#         self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")    
        
#     def forward(self, y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes):
#         total_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
#         total_weights = torch.tensor(0, dtype=torch.float32).to(self.device)

#         for y_img, y_exam, pred_img, pred_exam, size in zip(
#                 y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes
#             ):

#             exam_loss = self.bce(pred_exam, y_exam)
#             exam_loss = torch.sum(exam_loss * self.label_w, 1)[0]  
            
#             image_loss = self.bce(pred_img, y_img) 
#             qi = torch.sum(y_img) / size  # ratio of positives, not good for training
#             image_loss = torch.sum(self.img_w * qi * image_loss)

#             total_loss += exam_loss + image_loss
#             total_weights += self.label_w.sum() + self.img_w * qi * size

#             # total_loss += exam_loss
#             # total_weights += self.label_w.sum()

#         final_loss = total_loss / total_weights
#         return final_loss


class RSNAWLoss(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.device = torch.device("cuda") if cuda else torch.device("cpu")

        self.label_w = torch.tensor([
            0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 
            0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988
        ]).view(1, -1).to(self.device)

        self.img_w = 0.07361963
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self.bce = torch.nn.BCELoss(reduction="none")    
        
    def forward(self, y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes):
        total_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        total_weights = torch.tensor(0, dtype=torch.float32).to(self.device)

        for y_img, y_exam, pred_img, pred_exam, size in zip(
                y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes
            ):

            y_img = y_img[:len(pred_img)]

            exam_loss = self.bce(pred_exam, y_exam)
            exam_loss = torch.sum(exam_loss * self.label_w, 1)[0]  
            
            image_loss = self.bce(pred_img, y_img).mean()
            qi = torch.sum(y_img)
            image_loss = torch.sum(self.img_w * qi * image_loss)

            total_loss += exam_loss + image_loss
            total_weights += self.label_w.sum() + self.img_w * qi

            # break

        final_loss = total_loss / total_weights
        return final_loss