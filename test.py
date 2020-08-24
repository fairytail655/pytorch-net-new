import torch
from models import *

net = resnet20()

# for params in net.parameters():
    # data = params.data.numpy()
    # print(data.reshape(-1))
    # break

for name in net.state_dict():
    a = name
    data = net.state_dict()[a]
    print(a)
    print(data)
    break