import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  MyBinarizeLinear, MyBinarizeConv2d, MyBinarizeTanh

__all__ = ['resnet20_my', 'resnet20_my_1w1a']

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return MyBinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return MyBinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def linear(in_planes, out_planes):
    return MyBinarizeLinear(in_planes, out_planes, bias=False)


def act():
    return nn.Relu()


def act_1w1a():
    return MyBinarizeTanh()

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act2(out)

        return out


class ResNet_My(nn.Module):

    def __init__(self, num_classes=10, in_dim=3):
        super(ResNet_My, self).__init__()
        self.inflate = 4
        self.inplanes = 16*self.inflate
        self.conv1 = conv3x3(in_dim, self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act1 = act()
        self.layer1 = self._make_layer(BasicBlock, 16*self.inflate, 3)
        self.layer2 = self._make_layer(BasicBlock, 32*self.inflate, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64*self.inflate, 3, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = linear(64*self.inflate, num_classes)

        self._initialize_weights()

        self.train_config = {
            'cifar10': {
                # 'epochs': 200,
                # 'batch_size': 128,
                # 'opt_config': {
                #         0: {'optimizer': 'Adam', 'lr': 1e-2, 'weight_decay': 1e-4},
                #         30: {'lr': 5e-3},
                #         80: {'lr': 1e-3, 'weight_decay': 0},
                #         120: {'lr': 1e-4},
                #         150: {'lr': 1e-5},
                # },
                'epochs': 120,
                'batch_size': 128,
                'opt_config': {
                        0: {'optimizer': 'Adam', 'lr': 1e-3, 'weight_decay': 1e-4},
                        30: {'lr': 5e-4},
                        50: {'lr': 1e-4},
                        80: {'lr': 1e-5},
                        100: {'lr': 1e-6},
                },
                'transform': {
                    'train': 
                        transforms.Compose([
                            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
                    ]),
                    'eval': 
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                },    
            },
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MyBinarizeConv2d) or isinstance(m, MyBinarizeLinear):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # if isinstance(m, MyBinarizeConv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # elif isinstance(m, MyBinarizeLinear):
            #     m.weight.data.normal_(0, 0.01)
            #     if m.bias is not None:
            #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def set_value(self, epoch, epochs, is_training):
        v = torch.linspace(0, math.log(1000), epochs)[epoch].exp()
        for m in self.modules():
            if isinstance(m, MyBinarizeLinear) or isinstance(m, MyBinarizeLinear) or isinstance(m, MyBinarizeTanh):
                m.set_value(v, is_training)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock_1w1a(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_1w1a()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_1w1a()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act2(out)

        return out


class ResNet_My_1W1A(nn.Module):

    def __init__(self, num_classes=10, in_dim=3):
        super(ResNet_My_1W1A, self).__init__()
        self.inflate = 1
        self.inplanes = 16*self.inflate
        self.conv1 = conv3x3(in_dim, self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act1 = act_1w1a()
        self.layer1 = self._make_layer(BasicBlock_1w1a, 16*self.inflate, 3)
        self.layer2 = self._make_layer(BasicBlock_1w1a, 32*self.inflate, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock_1w1a, 64*self.inflate, 3, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = linear(64*self.inflate, num_classes)

        self._initialize_weights()

        self.train_config = {
            'cifar10': {
                'epochs': 100,
                'batch_size': 128,
                'opt_config': {
                    0: {'optimizer': 'Adam', 'lr': 1e-3, 'weight_decay': 1e-4},
                    20: {'lr': 5e-4},
                    40: {'lr': 1e-4},
                    60: {'lr': 1e-5},
                    80: {'lr': 1e-6},
                },
                # 'epochs': 120,
                # 'batch_size': 128,
                # 'opt_config': {
                #         0: {'optimizer': 'Adam', 'lr': 1e-2, 'weight_decay': 1e-4},
                #         30: {'lr': 5e-3},
                #         50: {'lr': 1e-3, 'weight_decay': 0},
                #         80: {'lr': 1e-4},
                #         100: {'lr': 1e-5},
                # },
                'transform': {
                    'train': 
                        transforms.Compose([
                            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
                    ]),
                    'eval': 
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                },    
            },
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MyBinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MyBinarizeLinear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def set_value(self, epoch, epochs, is_training):
        v = torch.linspace(0, math.log(1000), epochs)[epoch].exp()
        for m in self.modules():
            if isinstance(m, MyBinarizeLinear) or isinstance(m, MyBinarizeLinear) or isinstance(m, MyBinarizeTanh):
                m.set_value(v, is_training)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_my(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3

    return ResNet_My(num_classes=num_classes, in_dim=in_dim)


def resnet20_my_1w1a(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3

    return ResNet_My_1W1A(num_classes=num_classes, in_dim=in_dim)
