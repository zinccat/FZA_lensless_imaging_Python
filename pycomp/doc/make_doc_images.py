from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import compsense
import os

IMAGES_BASE = 'images'

def savefig(title):
    plt.savefig(os.path.join(IMAGES_BASE, title), bbox_inches='tight', pad_inches=0)


def tutorial1():
    P = compsense.problems.prob701(sigma=1e-3)

    tau = 0.00005

    alg = compsense.algorithms.TwIST(
        P,
        tau,
        stop_criterion=1,
        tolA=1e-3
    )
    x = alg.solve()

    y  = P.reconstruct(x)

    plt.figure()
    plt.imshow(P.signal, cmap=cm.gray, origin='lower')
    plt.title('Original Image')
    savefig('prob701_original.png')
    
    plt.figure()
    plt.imshow(P.b.reshape(P.A.out_signal_shape), cmap=cm.gray, origin='lower')
    plt.title('Blurred and Noisy Image')
    savefig('prob701_distorted.png')

    plt.figure()
    plt.imshow(y, cmap=cm.gray, origin='lower')
    plt.title('Reconstructed Image')
    savefig('prob701_reconstructed.png')

    plt.figure()
    plt.semilogy(alg.times, alg.objectives, lw=2)
    plt.title('Evolution of the objective function')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)
    savefig('prob701_objective.png')

    plt.figure()
    plt.semilogy(alg.times, alg.mses, lw=2)
    plt.title('Evolution of the mse')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)
    savefig('prob701_mse.png')



if __name__ == '__main__':
    tutorial1()    
    plt.show()