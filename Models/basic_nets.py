import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA




def get_basic_net(net, n_classes, input_size=None, input_channel=None):
    if net == "MLPNet":
        model = MLPNet(input_size, input_channel, n_classes)
    elif net == "LeNet":
        model = LeNet(input_size, input_channel, n_classes)
    elif net == "TFCNN":
        model = TFCNN(n_classes)
    elif net == "VGG8":
        model = VGG(8, n_classes, False)
    elif net == "VGG11":
        model = VGG(11, n_classes, False)
    elif net == "VGG13":
        model = VGG(13, n_classes, False)
    elif net == "VGG16":
        model = VGG(16, n_classes, False)
    elif net == "VGG19":
        model = VGG(19, n_classes, False)
    elif net == "VGG8-BN":
        model = VGG(8, n_classes, True)
    elif net == "VGG11-BN":
        model = VGG(11, n_classes, True)
    elif net == "VGG13-BN":
        model = VGG(13, n_classes, True)
    elif net == "VGG16-BN":
        model = VGG(16, n_classes, True)
    elif net == "VGG19-BN":
        model = VGG(19, n_classes, True)
    elif net == "ResNet8":
        model = ResNet(8, n_classes)
    elif net == "ResNet20":
        model = ResNet(20, n_classes)
    elif net == "ResNet32":
        model = ResNet(32, n_classes)
    elif net == "ResNet44":
        model = ResNet(44, n_classes)
    elif net == "ResNet56":
        model = ResNet(56, n_classes)
    else:
        raise ValueError("No such net: {}".format(net))



    return model


# for FedAws
def get_orth_weights(d, c):
    assert d > c, "d: {} must be larger than c: {}".format(d, c)
    xs = np.random.randn(d, d)
    pca = PCA(n_components=c)
    pca.fit(xs)

    # c \times d
    ws = pca.components_

    ws = torch.FloatTensor(ws)
    return ws


class ClassifyNet(nn.Module):
    def __init__(self, net, init_way, n_classes):
        super().__init__()
        self.net = net
        self.init_way = init_way
        self.n_classes = n_classes

        model = get_basic_net(net, n_classes)

        self.h_size = model.h_size

        self.encoder = model.encoder

        self.classifier = nn.Linear(
            self.h_size, self.n_classes, bias=False
        )

        if self.init_way == "orth":
            ws = get_orth_weights(self.h_size, self.n_classes)
            self.classifier.load_state_dict({"weight": ws})

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits


class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))


class MLPNet(nn.Module):
    def __init__(self, input_size, input_channel, n_classes=10):
        super().__init__()
        self.input_size = input_channel * input_size ** 2
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            Reshape(),
            nn.Linear(self.input_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
        )

        self.h_size = 128

        self.classifier = nn.Sequential(
            nn.Linear(128, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class LeNet(nn.Module):
    def __init__(self, input_size, input_channel, n_classes=10):
        super().__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        if self.input_size == 28:
            self.h_size = 16 * 4 * 4
        elif self.input_size == 32:
            self.h_size = 16 * 5 * 5
        else:
            raise ValueError("No such input_size.")

        self.classifier = nn.Sequential(
            nn.Linear(self.h_size, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        #return code, logits
        return logits


class TFCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        self.h_size = 64 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits

class ConvNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size = (32,32), dataset = 'cifar10'):
        super(ConvNet, self).__init__()
        channel_dict = {
            "cifar10": 3,
            "cinic10": 3,
            "cifar100": 3,
            "mnist": 1,
            "fmnist": 1,
            "femnist": 1,
        }
        channel =  channel_dict.get(dataset)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        print(f"num feat {num_feat}")
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.get_feature(x)
        out = self.classifier(out)
        return out

    def get_feature(self,x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat



class VGG(nn.Module):
    def __init__(
        self,
        n_layer=11,
        n_classes=10,
        use_bn=False,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.use_bn = use_bn

        self.cfg = self.get_vgg_cfg(n_layer)

        self.encoder = nn.Sequential(
            self.make_layers(self.cfg),
            Reshape(),
        )

        self.h_size = 512

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def get_vgg_cfg(self, n_layer):
        if n_layer == 8:
            cfg = [
                64, 'M',
                128, 'M',
                256, 'M',
                512, 'M',
                512, 'M'
            ]
        elif n_layer == 11:
            cfg = [
                64, 'M',
                128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ]
        elif n_layer == 13:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ]
        elif n_layer == 16:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 'M',
                512, 512, 512, 'M',
                512, 512, 512, 'M'
            ]
        elif n_layer == 19:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'
            ]
        return cfg

    def conv3x3(self, in_channel, out_channel):
        layer = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        return layer

    def make_layers(self, cfg, init_c=3):
        block = nn.ModuleList()

        in_c = init_c
        for e in cfg:
            if e == "M":
                block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                block.append(self.conv3x3(in_c, e))
                if self.use_bn is True:
                    block.append(nn.BatchNorm2d(e))
                block.append(nn.ReLU(inplace=True))
                in_c = e
        block = nn.Sequential(*block)
        return block

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ 6n + 2: 8, 14, 20, 26, 32, 38, 44, 50, 56
    """

    def __init__(self, n_layer=20, n_classes=10):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes

        conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        bn1 = nn.BatchNorm2d(16)

        assert ((n_layer - 2) % 6 == 0), "SmallResNet depth is 6n+2"
        n = int((n_layer - 2) / 6)

        self.cfg = (BasicBlock, (n, n, n))
        self.in_planes = 16

        layer1 = self._make_layer(
            block=self.cfg[0], planes=16, stride=1, num_blocks=self.cfg[1][0],
        )
        layer2 = self._make_layer(
            block=self.cfg[0], planes=32, stride=2, num_blocks=self.cfg[1][1],
        )
        layer3 = self._make_layer(
            block=self.cfg[0], planes=64, stride=2, num_blocks=self.cfg[1][2],
        )

        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(True),
            layer1,
            layer2,
            layer3,
            avgpool,
            Reshape(),
        )

        self.h_size = 64 * self.cfg[0].expansion

        self.classifier = nn.Linear(
            64 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(self, block, planes, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return logits,hs
