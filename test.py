import torch
import torch.nn as nn
import numpy as np
from models import *
from matplotlib import pyplot as plt

# net = vgg_my()
# a = torch.ones(1, 3, 32, 32)
# out = net(a)
# net = MyBinarizeLinear(1, 1, bias=False)
# net.v = 2
# a = torch.Tensor([0.2])
# out = net(a)
# out.backward()

# print(out)
# print(net.weight.grad)

a = torch.ones(2, 2, 2, 2)

for i in range(a.size(0)):
    for j in range(a.size(1)):
        b = a[i][j]

        print(b)
