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


def SelfBinarize(tensor, v, is_training=True):
    if is_training == False:
        return tensor.sign()
    else:
        return torch.tanh(v * tensor)


class SelfBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(SelfBinarizeLinear, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.v = 1

    def set_value(self, v, is_training):
        self.is_training = is_training
        self.v = v

    def forward(self, input):
        bw = SelfBinarize(self.weight, self.v, self.is_training)
        out = nn.functional.linear(input, bw, None)

        return out


class SelfBinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(SelfBinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.v = 1

    def set_value(self, v, is_training):
        self.v = v
        self.is_training = is_training

    def forward(self, input):
        bw = SelfBinarize(self.weight, self.v, self.is_training)
        out = nn.functional.conv2d(input, bw, None, self.stride, self.padding, self.dilation, self.groups)

        return out


class SelfTanh(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(SelfTanh, self).__init__(*kargs, **kwargs)
        self.is_training = True
        self.v = 1

    def set_value(self, v, is_training):
        self.v = v
        self.is_training = is_training

    def forward(self, input):

        return SelfBinarize(input, self.v, self.is_training)