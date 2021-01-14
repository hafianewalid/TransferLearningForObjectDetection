import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch


def pretrained(name):
    if 'resnet18' in name: return torchvision.models.resnet18(pretrained=True)
    if 'resnet34' in name: return torchvision.models.resnet34(pretrained=True),
    if 'resnet50' in name: return torchvision.models.resnet50(pretrained=True)
    if 'resnet101' in name: return torchvision.models.resnet101(pretrained=True)
    if 'resnet152' in name: return torchvision.models.resnet152(pretrained=True)
    if 'alexnet' in name: return torchvision.models.alexnet(pretrained=True)
    if 'squeezenet' in name: return torchvision.models.squeezenet1_0(pretrained=True)
    if 'vgg16' in name: return torchvision.models.vgg16(pretrained=True)
    if 'densenet' in name: return torchvision.models.densenet161(pretrained=True)
    if 'inception' in name: return torchvision.models.inception_v3(pretrained=True)
    if 'googlenet' in name: return torchvision.models.googlenet(pretrained=True)
    if 'shufflenet' in name: return torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    if 'mobilenet' in name: return torchvision.models.mobilenet_v2(pretrained=True)
    if 'mnasnet' in name: return torchvision.models.mnasnet1_0(pretrained=True)


class Feature_Extracting(nn.Module):
    def __init__(self, pretrained_name='resnet18'):
        super(Feature_Extracting, self).__init__()

        # self.model = torchvision.models.resnet152(pretrained=True)

        self.model = pretrained(pretrained_name)
        if (type(self.model) == tuple):
            self.model = self.model[0]
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        return self.feature_extractor(x)


class SingleBboxHead(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hiddenLayers=128):
        super(SingleBboxHead, self).__init__()

        self.head_bbox = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(num_features, hiddenLayers),
                                       nn.ReLU(),
                                       nn.LayerNorm(hiddenLayers),
                                       nn.Dropout(0.5),
                                       nn.Linear(hiddenLayers, 4),
                                       nn.Sigmoid())

        self.head_class = nn.Sequential(
            nn.Dropout(0.5),
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hiddenLayers),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hiddenLayers),
            nn.Linear(hiddenLayers, num_classes),
            nn.ReLU()
        )

    def forward(self, features):
        features = features.view(features.size()[0], -1)

        y_bbox = self.head_bbox(features)
        y_class = self.head_class(features)

        return y_bbox, y_class


class MultipleBboxHead(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, num_box: int):
        super(MultipleBboxHead, self).__init__()

        self.head_bbox = nn.Sequential(

            nn.Conv2d(num_channels, int(num_channels / 8), kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(),
            nn.BatchNorm2d(int(num_channels / 8)),
            nn.Dropout(p=0.2),

            nn.Conv2d(int(num_channels / 8), int(num_channels / 4), kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(int(num_channels / 4)),
            nn.Dropout(p=0.2),

            nn.Conv2d(int(num_channels / 4), int(num_channels / 2), kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(int(num_channels / 2)),
            nn.Dropout(p=0.2),

            nn.Conv2d(int(num_channels / 2), 4 * num_box, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout(p=0.2),
            nn.Sigmoid()

        )

        self.head_class = nn.Sequential(

            nn.Conv2d(num_channels, int(num_channels / 8), kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),

            nn.BatchNorm2d(int(num_channels / 8)),
            nn.Dropout(p=0.1),
            nn.Conv2d(int(num_channels / 8), int(num_channels / 4), kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),

            nn.BatchNorm2d(int(num_channels / 4)),
            nn.Dropout(p=0.1),
            nn.Conv2d(int(num_channels / 4), int(num_channels / 2), kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),

            nn.BatchNorm2d(int(num_channels / 2)),
            nn.Dropout(p=0.1),
            nn.Conv2d(int(num_channels / 2), num_box * (num_classes + 1), kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.Sigmoid()

        )

    def forward(self, features):
        y_bbox = self.head_bbox(features)

        y_class = self.head_class(features)

        return y_bbox, y_class[:, :-1, :, :], y_class[:, -1, :, :]
