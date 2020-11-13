import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from model_zoo.custom_forward import add_ft_extractor, add_ft_extractor_enet

RESNETS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def define_model(name, num_classes=1):
    # Load pretrained model
    if "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name in RESNETS:
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "efficientnet" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        raise NotImplementedError

    # Replace the last layer
    if "efficientnet" not in name:
        model.nb_ft = model.fc.in_features
        model.fc = nn.Linear(model.nb_ft, num_classes)
        add_ft_extractor(model)
    else:
        model.nb_ft = model._fc.in_features
        model._fc = nn.Linear(model.nb_ft, num_classes)
        add_ft_extractor_enet(model)

    return model
