"""
Base class for all algorithms.

.. codeauthor:: Amit Aides <amitibo@tx.technion.ac.il>

"""

from __future__ import division
import numpy as np
from ..problems import problemBase
import time


class algorithmBase(object):
    """
    Base class for algorithms

    Attributes
    ----------
    name : string
        Name of algorithm.
    P : instance of a subclass of problemBase
        The problem that the algorithm solves.
        
    Methods
    -------
    """

    def __init__(self, name, P):
        """
        Parameters
        ----------
        name : string
            Name of algorithm.
        P : instance of a subclass of problemBase
            The problem that the algorithm solves.
        """
        
        assert isinstance(P, problemBase), 'P should be an instanc of a subclass of problemBase'
            
        self._name = name
        self._P = P
        self._true_x = P.x0
        self._times = [time.time()]
        self._mses = []
        self._objectives = []
        
    @property
    def name(self):
        """Name of the algorithm
        """
        return self._name
        
    @property
    def P(self):
        """The problem that the algorithm is set to solve
        """
        return self._P
        
    def _calc_stats(self, x, objective):
        """Calculate statistics of the algorithm: mse, objective, time.
        Should be called internally by the algorithm.
        """
        
        self._times.append(time.time() - self._start_time)
        
        if not obj == None:
            self._objectives.append(obj)
            
        temp = self._true_x - x
        self._mses.append(np.sum(temp * temp)/x.size)
        
    @property
    def mses(self):
        """The statistics of the MSE per iteration.
        """
        return np.array(self._mses)

    @property
    def times(self):
        """Time per iteration.
        """
        return np.array(self._times)
    
    @property
    def objectives(self):
        """The statistics of the objective value per iteration.
        """
        return np.array(self._objectives)
    
    def _solve(self, x0):
        """Apply the operator on the input signal. Should be overwritten by the operator.
        This function is called by the `solve` method.
        
        Parameters
        ==========
        x0 : array
            Initial values for x.
        """
        
        raise NotImplementedError()
        
    def solve(self, x_init=None):
        """Solve the problem
        
        Parameters
        ----------
        x_init : array like, optional
            Initial value for x.
            
        Returns
        -------
        x : array,
            Solution of the main algorithm
        """
        
        return self._solve(x_init)
        