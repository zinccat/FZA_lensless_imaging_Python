"""
Solve the Missing Pixels problem using two approaches:

   * Wavelet basis as a sparsifying basis
   * Total Vatiation
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import compsense


def show_results(P, alg, x, algorithm_name):
    """Show graphs"""
    
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
    plt.title('Distorted Image')
    
    plt.figure()
    plt.imshow(y, cmap=cm.gray, origin='lower')
    plt.title('Reconstructed Image using %s' % algorithm_name)
    
    plt.figure()
    plt.semilogy(alg.times, alg.objectives, lw=2)
    plt.title('Evolution of the objective function (%s)' % algorithm_name)
    plt.xlabel('CPU time (sec)')
    plt.grid(True)
    
    plt.figure()
    plt.semilogy(alg.times, alg.mses, lw=2)
    plt.title('Evolution of the mse (%s)' % algorithm_name)
    plt.xlabel('CPU time (sec)')
    plt.grid(True)


def solve_wavelet():
    """Solve using sparsifying wavelet basis"""

    #
    # Generate environment
    #
    P = compsense.problems.probMissingPixels(
        fill_ratio=0.9,
        wavelet='bior3.1',
        sigma=1e-3
    )
  
    #
    # Regularization parameter
    #
    tau = 0.00005

    #
    # Solve an L1 recovery problem:
    # minimize  1/2|| Ax - b ||_2^2  +  \tau ||x||_1
    #
    alg = compsense.algorithms.TwIST(
        P,
        tau,
        stop_criterion=1,
        tolA=1e-3
        )
    x = alg.solve()
    
    return P, alg, x

    
def solve_TV():
    """Solve using TV algorithm"""

    from skimage.filter import tv_denoise

    def Psi(x, threshold):
        """
        Deblurring operator.
        
        Arguments:
        ----------
        x : array-like, shape = [m, n]
            Estimated signal
           
        threshold : float
            Threshold for the deblurring algorithm
        """
    
        img_estimated = tv_denoise(x, weight=threshold/2, n_iter_max=4)
    
        return img_estimated
    
        
    def Phi(x):
        """
        Regularization operator.
        
        Arguments:
        ----------
        x : array-like, shape = [m, n]
            Input signal vector.
        """
    
        dy = np.zeros_like(x[:, :])
        dx = np.zeros_like(x[:, :])
        
        dy[:-1] = np.diff(x[:, :], axis=0)
        dx[:, :-1] = np.diff(x[:, :], axis=1)
        phi = np.sum(np.sqrt(dy**2 + dx**2)) 
            
        return phi

    #
    # Generate environment
    #
    P = compsense.problems.probMissingPixels(wavelet=None, sigma=1e-3)
  
    #
    # Regularization parameter
    #
    tau = 0.1
    tolA = 1e-8
    
    #
    # Solve an L1 recovery problem:
    # minimize  1/2|| Ax - b ||_2^2  +  \tau || \nabla x ||_1
    #
    alg = compsense.algorithms.TwIST(
        P,
        tau,
        psi_function=Psi,
        phi_function=Phi,
        tolA=tolA
        )
    x = alg.solve()
   
    return P, alg, x


if __name__ == '__main__':
    
    P, alg, x = solve_wavelet()
    show_results(P, alg, x, 'Wavelet basis')
    P, alg, x = solve_TV()
    show_results(P, alg, x, 'Total Variation')

    plt.show()

