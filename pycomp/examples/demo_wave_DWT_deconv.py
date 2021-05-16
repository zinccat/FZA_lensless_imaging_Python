"""
Orthogonal wavelet based image deburring using TwIST:
Setting: cameraman, blur uniform 9*9.

This demo illustrates the computation of  

    xe = arg min 0.5 ||Ax-y||^2 + tau ||x||_1
            x

with the TwIST algorithm, where the operator A= H o W is the  
composition of the blur and DWT synthesis operator.

For further details about the TwIST algorithm, see the paper:

J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step
Iterative Shrinkage/Thresholding Algorithms for Image 
Restoration",  IEEE Transactions on Image processing, 2007.

(available at   http://www.lx.it.pt/~bioucas/publications.html)


Authors of original Matlab demo Jose Bioucas-Dias and Mario Figueiredo, 
Instituto Superior Tecnico, October, 2007
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import compsense


def main():
    """
    Main Function
    """

    #
    # Generate environment
    #
    P = compsense.problems.prob701(sigma=1e-3, undecimated=False)
  
    #
    # Regularization parameter
    #
    tau = 0.00005

    #
    # Solve an L1 recovery problem:
    # minimize  1/2|| Ax - b ||_2^2  +  tau ||x||_1
    #
    alg = compsense.algorithms.TwIST(
        P,
        tau,
        stop_criterion=1,
        tolA=1e-3
        )
    x = alg.solve()
    
    #
    # The solution x is the reconstructed signal in the sparsity basis.
    # Use the function handle P.reconstruct to use the coefficients in
    # x to reconstruct the original signal.
    #
    y  = P.reconstruct(x)

    #
    # Show results
    #
    plt.figure()
    plt.imshow(P.signal, cmap=cm.gray, origin='lower')
    plt.title('Original Image')

    plt.figure()
    plt.imshow(P.b.reshape(P.A.out_signal_shape), cmap=cm.gray, origin='lower')
    plt.title('Blurred and Noisy Image')
    
    plt.figure()
    plt.imshow(y, cmap=cm.gray, origin='lower')
    plt.title('Reconstructed Image')
    
    plt.figure()
    plt.semilogy(alg.times, alg.objectives, lw=2)
    plt.title('Evolution of the objective function')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)
    
    plt.figure()
    plt.semilogy(alg.times, alg.mses, lw=2)
    plt.title('Evolution of the mse')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)

    plt.show()

    
if __name__ == '__main__':    
    main(True)
