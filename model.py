import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn.parameter import Parameter

# ECA模块
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Encoder1(nn.Module):
    def __init__(self, bands, feature_dim):
        super(Encoder1, self).__init__()
        self.dim1 = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, self.dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim1 * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim1 * 2, self.dim1 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim1 * 4, self.dim1 * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.conv1(x)  # [90, 64, 12, 12]
        x2 = self.conv2(x1)  # [90, 128, 6, 6]
        x3 = self.conv3(x2)  # [90, 256, 3, 3]
        return x3


class Encoder11(nn.Module):
    def __init__(self, bands, feature_dim):
        super(Encoder11, self).__init__()
        self.dim1 = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, self.dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            eca_layer(self.dim1),
            ASPP(self.dim1, [1, 3, 5], self.dim1),
            nn.Conv2d(self.dim1, self.dim1 * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            eca_layer(self.dim1 * 2),
            nn.Conv2d(self.dim1 * 2, self.dim1 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            eca_layer(self.dim1 * 4),
            nn.Conv2d(self.dim1 * 4, self.dim1 * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.conv1(x)  # [90, 64, 12, 12]
        x2 = self.conv2(x1)  # [90, 128, 6, 6]
        x3 = self.conv3(x2)  # [90, 256, 3, 3]
        return x3


class Projector(nn.Module):
    def __init__(self, low_dim):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(low_dim*2, 16)
        self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        # self.relu_mlp = nn.ReLU()
        self.fc2 = nn.Linear(16, low_dim)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu_mlp(x)
        x = self.fc2(x)
        x = self.l2norm(x)

        return x


class Classifier0(nn.Module):
    def __init__(self, n_classes):
        super(Classifier0, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.conv(x)
        # x1 = x.view(x.size(0), -1)
        # x = self.avg(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        # x = torch.softmax(x, dim=1)
        return x


class Supervisednetwork(nn.Module):
    def __init__(self, bands, n_classes, low_dim):
        super(Supervisednetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(low_dim*2, low_dim*2, 1),
            nn.BatchNorm2d(low_dim*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.encoder = Encoder1(bands, low_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(low_dim*2, n_classes)

    def forward(self, x):
        feature = self.encoder(x)
        x = self.conv(feature)
        # x = self.avgpool(feature)
        x = torch.flatten(x, start_dim=1)
        y = self.head(x)
        return y


class Classifier(nn.Module):

    def __init__(self, num_classes=10, feature_dim=256):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.features_dim = feature_dim
        # pseudo head and worst-case estimation head
        mlp_dim = 2 * self.features_dim
        self.head = nn.Linear(self.features_dim, num_classes)
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            # nn.Dropout(0.5),
            nn.Linear(mlp_dim, num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        y = self.head(x1)
        y_pseudo = self.pseudo_head(x1)
        # y_pseudo = self.head(x1)
        return y, y_pseudo
        # return y


class Classifier11(nn.Module):

    def __init__(self, num_classes=10, feature_dim=256):
        super(Classifier11, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.features_dim = feature_dim
        # pseudo head and worst-case estimation head
        mlp_dim = 2 * self.features_dim
        self.head = nn.Linear(self.features_dim, num_classes)
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            # nn.Dropout(0.5),
            nn.Linear(mlp_dim, num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        y = self.head(x1)
        # y_pseudo = self.pseudo_head(x1)
        # y_pseudo = self.head(x1)
        return y, y
        # return y



class Network2(nn.Module):
    def __init__(self, bands, n_classes, low_dim):
        super(Network2, self).__init__()

        self.encoder = Encoder1(bands, low_dim)
        self.projector = Projector(low_dim)
        self.classifier = Classifier0(n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, u_w=None, u_s=None):
        if u_w==None and u_s==None :
            feature = self.encoder(x)
            cx = self.classifier(feature)
            feature = torch.mean(feature, dim=(2, 3))
            feature = self.projector(feature)
            return feature, cx

        feature1_3 = self.encoder(x)
        cx = self.classifier(feature1_3)
        feature1_3 = self.avgpool(feature1_3)
        feature1_3 = torch.mean(feature1_3, dim=(2, 3))
        feature1_3 = self.projector(feature1_3)

        feature2_3 = self.encoder(u_w)
        cuw = self.classifier(feature2_3)
        feature2_3 = self.avgpool(feature2_3)
        feature2_3 = torch.mean(feature2_3, dim=(2, 3))
        feature2_3 = self.projector(feature2_3)

        feature3_3 = self.encoder(u_s)
        cus = self.classifier(feature3_3)
        feature3_3 = self.avgpool(feature3_3)
        feature3_3 = torch.mean(feature3_3, dim=(2, 3))
        feature3_3 = self.projector(feature3_3)

        return feature1_3, feature2_3, feature3_3, cx, cuw, cus


class Network(nn.Module):
    def __init__(self, bands, n_classes, feature_dim):
        super(Network, self).__init__()
        self.encoder = Encoder1(bands, feature_dim)
        self.classifier = Classifier(n_classes, feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.encoder(x)
        y, y_pseudo = self.classifier(f)

        return y, y_pseudo
        # return y

class Network_Unimatch(nn.Module):
    def __init__(self, bands, n_classes, feature_dim):
        super(Network_Unimatch, self).__init__()
        self.encoder = Encoder1(bands, feature_dim)
        self.classifier = Classifier11(n_classes, feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, infer=False):
        f = self.encoder(x)
        if not infer:
            batch_size = x.shape[0] // 4
            fu, _, _ = f[batch_size:].chunk(3, dim=0)
            fp = F.dropout(fu, 0.05)
            f = torch.cat([f, fp], dim=0)
        y, y_pseudo = self.classifier(f)

        return y, y_pseudo


class Network_Unimatch_Improve(nn.Module):
    def __init__(self, bands, n_classes, feature_dim):
        super(Network_Unimatch_Improve, self).__init__()
        self.encoder = Encoder11(bands, feature_dim)
        self.classifier = Classifier11(n_classes, feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, infer=False):
        f = self.encoder(x)
        if not infer:
            batch_size = x.shape[0] // 4
            fu, _, _ = f[batch_size:].chunk(3, dim=0)
            fp = F.dropout(fu, 0.05)
            f = torch.cat([f, fp], dim=0)
        y, y_pseudo = self.classifier(f)

        return y, y_pseudo

class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance
