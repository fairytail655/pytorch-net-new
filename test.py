import torch
import torch.nn as nn
import numpy as np
from models import *
from matplotlib import pyplot as plt

# k = 2
# m = 1
# x = np.arange(-5, 5, 0.01, dtype='float32')
# x_t = torch.from_numpy(x)
# y_t = torch.nn.functional.hardtanh((x_t-m)*k)
# y = y_t.numpy()

# plt.plot(x, y)
# plt.show()
# x = torch.linspace(0, math.log(1000), 10).requires_grad_(True)

y = []
z = 100
for epoch in range(z):
    v = torch.linspace(0, math.log(1000), z)[epoch].exp()
    y.append(v.numpy())

plt.plot([i for i in range(z)], y)
plt.show()
