import numpy as np
import cv2
from interp2 import interp2
from scipy import interpolate


def pinhole(O, di, x, y, z, Lx, dp, Nx):
    if O.shape[2] == 3:
        O = cv2.cvtColor(O, cv2.COLOR_BGR2GRAY)
    m, n = O.shape
    Ly = m * Lx/n  # object height, unit: mm
    M = di/z  # magnification
    Lxi = M*Lx  # image width
    Lyi = M*Ly
    xi = M*x
    yi = M*y
    ds = Lxi/n
    Ny = Nx
    W = Nx*dp  # sensor size
    H = Ny*dp
    X = np.linspace(xi-Lxi/2, xi+Lxi/2-ds, n)
    Y = np.linspace(yi+Lyi/2-ds, yi-Lyi/2, m)
    Xq = np.linspace(-W/2, W/2-dp, Nx)
    Yq = np.linspace(H/2-dp, -H/2, Ny)
    f = interpolate.interp2d(X, Y, O, kind='linear', fill_value=0)
    return f(Xq, Yq)
