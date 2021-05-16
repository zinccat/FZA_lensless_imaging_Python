from numpy.fft import *
import numpy as np


def MyForwardOperatorPropagation(obj, H):
    FO = fftshift(fft2(fftshift(obj)))
    I = fftshift(ifft2(fftshift(FO*H)))
    I = np.real(I)
    return I
