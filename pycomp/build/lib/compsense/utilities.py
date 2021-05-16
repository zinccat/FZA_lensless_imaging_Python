"""
utilities - Several utilities useful when using pycompsesne
===========================================================

.. codeauthor:: Amit Aides <amitibo@tx.technion.ac.il>
"""

from __future__ import division
import pkg_resources
import numpy as np
import types    


def getResourcePath(name):
    """
    Return the path to a resource
    """

    return pkg_resources.resource_filename(__name__, "data/%s" % name)
    

def isFunction(f):
    """
    Check if object is a function.
    """

    return isinstance(f, types.FunctionType) or isinstance(f, types.MethodType) or hasattr(f, '__call__')


def softThreshold(x, threshold):
    """
    Apply Soft Thresholding
    
    Parameters
    ----------
    
    x : array-like
        Vector to which the soft thresholding is applied
        
    threshold : float
        Threhold of the soft thresholding
    
    Returns:
    --------
    y : array
        Result of the applying soft thresholding to x.
        
        .. math::
              
            y = sign(x) \star \max(\abs(x)-threshold, 0)
    """
    
    y = np.abs(x) - threshold
    y[y<0] = 0
    y[x<0] = -y[x<0]
    
    return y


def hardThreshold(x, threshold):
    """
    Apply Hard Thresholding
    
    Parameters
    ----------
    
    x : array-like
        Vector to which the hard thresholding is applied
        
    threshold : float
        Threhold of the hard thresholding
    
    Returns:
    --------
    y : array
        Result of the applying hard thresholding to x.
        
        .. math::
              
            y = x * (\abs(x) > threshold)
    """
    
    y = np.zeros_like(x)
    ind = np.abs(x) > threshold
    y[ind] = x[ind]
    
    return y

    