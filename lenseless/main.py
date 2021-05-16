# %%
import numpy as np
import cv2
from pinhole import pinhole
from FZA import FZA
from scipy import signal
from matplotlib import pyplot as plt
from MyAdjointOperatorPropagation import MyAdjointOperatorPropagation
from MyForwardOperatorPropagation import MyForwardOperatorPropagation
from TVnorm import TVnorm
from tvdenoise import tvdenoise
import compsense
import time

start_time=time.time()

# %%
di = 3  # the distance from mask to sensor
z1 = 20
x1 = 0
y1 = 0
Lx1 = 20  # object size
dp = 0.01  # pixel pitch
Nx = 256  # pixel numbers
Ny = 256
# %%
img = cv2.imread('./lena.bmp')
Im = pinhole(img, di, x1, y1, z1, Lx1, dp, Nx)/255
# cv2.imwrite('./l1.png', im)
# %%
S = 2*dp*Nx  # aperture diameter
r1 = 0.23  # FZA constant

M = di/z1
ri = (1+M)*r1

mask = FZA(S, 2*Nx, ri)  # generate the FZA mask *255
I = signal.convolve2d(Im, mask, mode='same')*2*dp*dp/(ri*ri)
I -= np.mean(I)
# plt.imshow(I, extent=[0, 1, 0, 1], cmap='gray')
# plt.show()
# %%
fu_max = 0.5 / dp
fv_max = 0.5 / dp
du = 2*fu_max / (Nx)
dv = 2*fv_max / (Ny)
[u, v] = np.meshgrid(np.arange(-fu_max, fu_max, du),
                     np.arange(-fv_max, fv_max, dv))
H = 1j*np.exp(-1j*(np.pi*(ri*ri))*(u*u + v*v))  # fresnel transfer function
Or = MyAdjointOperatorPropagation(I, H)
# plt.imshow(Or, extent=[0, 1, 0, 1], cmap='gray')
# plt.show()
# %%

tau = 0.005
tolA = 1e-6
iterations = 500
tv_iters = 2


def Psi(tv_iters):
    return lambda x, th: tvdenoise(x, 2/th, tv_iters)


def Forward(H):
    return lambda x: MyForwardOperatorPropagation(x, H)


def Adjoint(H):
    return lambda I: MyAdjointOperatorPropagation(I, H)


# %%
x = compsense.algorithms.TwIST_raw(I, Forward(H), tau, AT=Adjoint(H), psi_function=Psi(tv_iters), phi_function=TVnorm,
                                   stop_criterion=1, tolA=tolA, maxiter=iterations, miniter=iterations, enforce_monotone=True, init=2, verbose=True)
print(time.time()-start_time)
plt.imshow(x[0], extent=[0, 1, 0, 1], cmap='gray')
plt.savefig('lenaout.png')
plt.show()


# %%
