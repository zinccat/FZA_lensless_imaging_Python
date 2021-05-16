from numpy.fft import *
import numpy as np


def MyAdjointOperatorPropagation(I, H):
    # print(I.shape)
    FI = fftshift(fft2(fftshift(I)))
    # print(FI.shape)
    # Or = fftshift(ifft2(fftshift(FI.*conj(H))))
    # print(H.shape)
    Or = fftshift(ifft2(fftshift(FI/H)))

    Or = np.real(Or)
    return Or
