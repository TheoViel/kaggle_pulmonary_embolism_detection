import torch


def add_ft_extractor(resnet):
    resnet.extract_ft = lambda x: forward_extract_ft(resnet, x)


def forward_extract_ft(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    ft = torch.flatten(x, 1)

    x = self.fc(ft)

    return x.view(-1), ft


def add_ft_extractor_enet(effnet):
    effnet.extract_ft = lambda x: forward_extract_ft_enet(effnet, x)


def forward_extract_ft_enet(self, x):
    x = self.extract_features(x)

    x = self._avg_pooling(x)

    ft = x.flatten(start_dim=1)
    x = self._dropout(ft)
    x = self._fc(x)

    return x.view(-1), ft
