import numpy as np
from conv2c import conv2c


def diffh(x):
    h = np.array([0, 1, -1])
    sol = conv2c(x, h)
    return sol
