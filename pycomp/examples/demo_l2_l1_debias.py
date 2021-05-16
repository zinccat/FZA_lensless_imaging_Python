"""
This demo illustrates  the TwIST algorithm in
the l2-l1 optimization problem 

     xe = arg min 0.5*||A x-y||^2 + tau ||x||_1
             x

where A is a generic matrix and ||.||_1 is the l1 norm.
After obtaining the solution we implement a debias phase
 
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
    """Main Function"""
    
    #
    # signal length 
    #
    n = 4096
    
    #
    # observation length 
    #
    k = 1024

    #
    # number of spikes
    #
    n_spikes = 160
    
    #
    # random +/- 1 signal
    #
    x = np.zeros((n, 1))
    q = np.random.permutation(n)
    x.ravel()[q[:n_spikes]] = np.sign(np.random.randn(n_spikes))

    #
    # measurement matrix
    #
    print 'Building measurement matrix...'
    R = np.random.randn(k, n)
    
    #
    # normalize R (not necessary)
    #
    #u, s, v = np.linalg.svd(R)
    #R = R / s[0]
    print 'Finished creating matrix'

    #
    # noise variance
    #
    sigma = 1e-2
   
    #
    # observed data
    #
    y = np.dot(R, x) + sigma*np.random.randn(k, 1)
    
    #
    # Create the problem
    #
    P = compsense.problems.probCustom(A=R, b=y, x0=x)
    
    #
    # Set the TwIST algorithm
    #
    tau = 0.1*np.max(np.abs(np.dot(R.T, y)))
    
    alg = compsense.algorithms.TwIST(
        P,
        tau,
        stop_criterion=1,
        debias=True,
        tolA=1e-4,
        lam1=1e-3
        )
    x_twist = alg.solve()

    plt.figure()

    plt.subplot(211)
    plt.semilogy(alg.times[:alg.debias_start], alg.objectives[:alg.debias_start], 'r', lw=2)
    plt.semilogy(alg.times[alg.debias_start:], alg.objectives[alg.debias_start:], 'b', lw=2)
    plt.legend(('TwIST', 'Debias phase'))
    plt.title('lambda = %2.1e, sigma = %2.1e' % (tau,sigma))
    plt.xlabel('CPU time (sec)')
    plt.ylabel('Obj. function')
    plt.grid(which='both', axis='both')
    plt.subplot(212)
    plt.plot(x_twist, 'b', lw=2)
    plt.plot(x+2.5, 'k', lw=2)
    plt.legend(('TwIST','Original'))
    plt.title('TwIST MSE = %2.1e' % (np.sum((x_twist-x)**2)/x.size))

    plt.show()

    
if __name__ == '__main__':
    main()
