import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  SelfBinarizeConv2d, SelfBinarizeLinear, SelfTanh

__all__ = ['vgg_selfbinaring']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SelfBinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def linear(in_planes, out_planes):
    return SelfBinarizeLinear(in_planes, out_planes, bias=False)

def nonlinear():
    return nn.ReLU(inplace=True)

class VGG(nn.Module):

    def __init__(self, num_classes=10, in_dim=3):
        super(VGG, self).__init__()
        self.conv1 = conv3x3(in_dim, 128)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nonlinear()

        self.conv2 = conv3x3(128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = conv3x3(128, 256)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = conv3x3(256, 256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = conv3x3(256, 512)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = conv3x3(512, 512)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(512)

        self.fc = linear(512*4*4, num_classes)

        self._initialize_weights()
        self.epoch = 0
        self.epochs = 0
        self.is_training = True

        self.train_config = {
            'cifar10': {
                'epochs': 200,
                'batch_size': 128,
                'opt_config': {
                    0: {'optimizer': 'Adam', 'lr': 1e-3, 'weight_decay': 1e-4},
                    50: {'lr': 5e-4},
                    100: {'lr': 1e-4},
                    150: {'lr': 1e-5},
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
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_value(self, epoch, epochs, is_training):
        self.epochs = epochs
        self.epoch = epoch
        self.is_training = is_training

    def forward(self, x):
        v = torch.linspace(0, math.log(1000), self.epochs)[self.epoch].exp()
        for m in self.modules():
            if isinstance(m, SelfBinarizeConv2d):
                m.set_value(v, self.is_training)
            elif isinstance(m, SelfBinarizeLinear):
                m.set_value(v, self.is_training)
            elif isinstance(m, SelfTanh):
                m.set_value(v, self.is_training)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        x = self.conv6(x)
        x = self.maxpool6(x)
        x = self.bn6(x)
        x = self.nonlinear(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def vgg_selfbinaring(**kwargs):
    datasets = kwargs.get('dataset', 'cifar10')
    if datasets == 'mnist':
        num_classes = 10
        in_dim = 1
    elif datasets == 'cifar10':
        num_classes = 10
        in_dim = 3
    return VGG(num_classes, in_dim)
