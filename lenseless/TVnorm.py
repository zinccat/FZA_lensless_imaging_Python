import numpy as np
from diffh import diffh
from diffv import diffv


def TVnorm(x):
    # Nx, Ny, Nz = x.shape #三维
    Nx, Ny = x.shape
    if x.ndim > 2:
        Nz = x.shape[2]
    else:
        Nz = 1
    x = x.reshape(Nx, Ny*Nz)
    y = np.sum(np.sqrt(diffh(x)**2+diffv(x)**2))
    return y
