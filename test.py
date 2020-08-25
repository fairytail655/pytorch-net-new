import torch
from models import *

net = vgg()

# for params in net.parameters():
    # data = params.data.numpy()
    # print(data.reshape(-1))
    # break

for name in net.state_dict():
    a = name
    print(a)