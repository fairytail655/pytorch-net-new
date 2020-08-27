import torch
import torch.nn as nn
import numpy as np
from models import *
from matplotlib import pyplot as plt

# a = torch.ones(1, 3, 32, 32)
net = resnet20_my()
# b = net(a)

# for name in net.state_dict():
#     print(name)