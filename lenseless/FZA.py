import numpy as np
import cv2


def FZA(S, N, r1):
    [x, y] = np.meshgrid(np.linspace(-S/2, S/2-S/N, N),
                         np.linspace(-S/2, S/2-S/N, N))
    r_2 = x*x + y*y
    mask = 0.5*(1 + np.cos(np.pi*r_2/(r1**2)))
    return mask
