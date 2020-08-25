import torch
import pdb
import torch.nn as nn
import math

def Binarize(tensor):

    return tensor.sign()


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input.data = Binarize(input.data)

        temp = self.weight.data
        self.weight.data = Binarize(temp)
        out = nn.functional.linear(input, self.weight, None)
        self.weight.data = temp

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)

        temp = self.weight.data
        self.weight.data = Binarize(temp)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        self.weight.data = temp

        return out

class BinarizeLinear_1w32a(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear_1w32a, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        temp = self.weight.data
        self.weight.data = Binarize(temp)
        out = nn.functional.linear(input, self.weight, None)
        self.weight.data = temp

        return out


class BinarizeConv2d_1w32a(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d_1w32a, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        temp = self.weight.data
        self.weight.data = Binarize(temp)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        self.weight.data = temp

        return out


def SelfBinarize(tensor, epoch, epochs, is_training=True):
    if is_training == False:
        return tensor.sign()
    else:
        v = torch.linspace(0, math.log(1000), epochs)[epoch].exp()
        return nn.functional.tanh(v * tensor)


class SelfBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(SelfBinarizeLinear, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.epoch = 0

    def forward(self, input):
        bw = SelfBinarize(self.weight, self.epoch, self.is_training)
        out = nn.functional.linear(input, bw)

        return out


class SelfBinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(SelfBinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.epoch = 0

    def forward(self, input):
        bw = SelfBinarize(self.weight, self.epoch, self.is_training)
        out = nn.functional.conv2d(input, bw, None, self.stride, self.padding, self.dilation, self.groups)

        return out


class SelfTanh(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(SelfTanh, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.epoch = 0

    def forward(self, input):

        return SelfBinarize(input, self.epoch, self.is_training)