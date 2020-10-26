import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, ft_dim=2048, lstm_dim=64, dense_dim=256, logit_dim=64, use_msd=False, num_classes=9):
        super().__init__()
        self.use_msd = use_msd
        
        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim * 2),
            nn.ReLU(),
            nn.Linear(dense_dim * 2, dense_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
        
        self.logits_exam = nn.Sequential(
            nn.Linear(lstm_dim * 4 + dense_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )
        
        self.logits_img = nn.Sequential(
            nn.Linear(lstm_dim *  2 + dense_dim, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, 1),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
    
    def forward(self, x, p=None):
        features = self.mlp(x)
        features2, _ = self.lstm(features)

        features = torch.cat([features, features2], -1)
        
        mean = features.mean(1)
        max_, _ = features.max(1)
        pooled = torch.cat([mean, max_], -1)
        
        if self.use_msd and self.training:
            logits_exam = torch.mean(
                torch.stack(
                    [self.logits_exam(self.high_dropout(pooled)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
            
            logits_img = torch.mean(
                torch.stack(
                    [self.logits_img(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
        else:
            logits_exam = self.logits_exam(pooled)
            logits_img = self.logits_img(features)

        return logits_exam, logits_img.squeeze(-1)

