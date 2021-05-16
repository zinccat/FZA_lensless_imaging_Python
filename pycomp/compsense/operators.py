"""
operators
=========

The classes defined in this module, implement different operators that
operate on input signals. These operators are used for defining
problems. opBase should be subclassed for creating new operators.

..
    This module is based on MATLAB SPARCO Toolbox.
    Copyright 2008, Ewout van den Berg and Michael P. Friedlander
    http://www.cs.ubc.ca/labs/scl/sparco

.. codeauthor:: Amit Aides <amitibo@tx.technion.ac.il>

"""

from __future__ import division
import numpy as np
import numpy.fft as npfft
import scipy.fftpack as spfft


class opBase(object):
    """
    Base class for operators

    Attributes
    ----------
    name : string
        Name of operator.
    shape : (int, int)
        The shape of the operator.
    in_signal_shape : tuple of ints
        The shape of the input signal.
    out_signal_shape : tuple of ints
        The shape of the output signal.
    T : type(self)
        The transpose of the operator.

    Methods
    -------
    """

    def __init__(self, name, shape, in_signal_shape=None, out_signal_shape=None):
        """
        Parameters
        ----------
        name : string
            Name of operator.
        shape : (int, int)
            The shape of the operator. `shape[1]` is the size of the input signal.
            `shape[0]` is the size of the output signal.
        in_signal_shape : tuple of integers, optional (default=None)
            The shape of the input signal. The product of `in_signal_shape` should
            be equal to `shape[1]`. If `None`, then it is set to (shape[1], 1).
        out_signal_shape : tuple of ints
            The shape of the output signal. The product of `out_signal_shape` should
            be equal to `shape[1]`. If `in_signal_shape=None`, then it is set to
            (shape[0], 1). If `out_signal_shape=None` and `shape[0]=shape[1]` then
            `out_signal_shape=in_signal_shape`.
        """

        if in_signal_shape == None:
            in_signal_shape = (shape[1], 1)
            out_signal_shape = (shape[0], 1)
        elif out_signal_shape == None:
            if shape[0] == shape[1]:
                out_signal_shape = in_signal_shape
            else:
                out_signal_shape = (shape[0], 1)

        assert np.prod(
            in_signal_shape) == shape[1], 'Input signal shape does not conform to the shape of the operator'
        assert np.prod(
            out_signal_shape) == shape[0], 'Output signal shape does not conform to the shape of the operator'

        self._name = name
        self._shape = shape
        self._in_signal_shape = in_signal_shape
        self._out_signal_shape = out_signal_shape
        self._conj = False

    @property
    def name(self):
        """Name of operator.
        """
        return self._name

    @property
    def shape(self):
        """The shape of the operator.
        """
        if self._conj:
            return self._shape[::-1]
        else:
            return self._shape

    @property
    def in_signal_shape(self):
        """The shape of the input signal for the operator.
        """
        if self._conj:
            return self._out_signal_shape
        else:
            return self._in_signal_shape

    @property
    def out_signal_shape(self):
        """The shape of the output signal for the operator.
        """
        if self._conj:
            return self._in_signal_shape
        else:
            return self._out_signal_shape

    @property
    def T(self):
        """The transpose of the operator.
        """
        import copy

        new_copy = copy.copy(self)
        new_copy._conj = True
        return new_copy

    def _checkDimensions(self, x):
        """Check that the size of the input signal is correct.
        This function is called by the `__call__` method.

        Parameters
        ==========
        x : array
            Input signal in columnstack order.
        """

        if x.shape == (1, 1) and self._shape != (1, 1):
            raise Exception('Operator-scalar multiplication not yet supported')

        if x.shape[0] != self.shape[1]:
            raise Exception('Incompatible dimensions')

        if x.shape[1] != 1:
            raise Exception('Operator-matrix multiplication not yet supported')

    def _apply(self, x):
        """Apply the operator on the input signal. Should be overwritten by the operator.
        This function is called by the `__call__` method.

        Parameters
        ==========
        x : array
            Input signal in columnstack order.
        """

        raise NotImplementedError()

    def __call__(self, x):

        x = x.reshape((-1, 1))

        self._checkDimensions(x)

        return self._apply(x).reshape(self.out_signal_shape)


class opMatrix(opBase):
    """
    Operator that wraps a simple matrix.
    """

    def __init__(self, A):
        """
        Parameters
        ----------
        A : array like, [m, n]
            Matrix of dimension m, n.
        """

        try:
            self._A = np.array(A)
        except:
            raise Exception('Parameter A must be array like object')

        assert self._A.ndim == 2, "opMatrix supports only 2D matrices"
        m, n = self._A.shape

        super(opMatrix, self).__init__(
            name='Matrix',
            shape=(m, n),
            in_signal_shape=(n, 1),
            out_signal_shape=(m, 1)
        )

    def _apply(self, x):

        if self._conj:
            y = np.dot(self._A.T, x)
        else:
            y = np.dot(self._A, x)

        return y


class opBlur(opBase):
    """
    Two-dimensional blurring operator. creates a blurring operator
    for M by N images. This function is used for the GPSR-based test
    problems and is based on the implementation by Figueiredo, Nowak 
    and Wright, 2007.

    Parameters
    ----------
    shape : (int, int)
        Shape of target images.
    """

    def __init__(self, shape):

        assert len(shape) == 2, "opBlur supports operations on 2D matrices only"
        m, n = shape
        size = m * n

        super(opBlur, self).__init__(
            name='Blur',
            shape=(size, size),
            in_signal_shape=shape
        )

        yc = int(m/2 + 1)
        xc = int(n/2 + 1)

        #
        # Create blurring mask
        #
        h = np.zeros((m, n))
        g = np.arange(-4, 5)
        for i in g:
            h[i+yc, g+xc] = 1 / (1 + i*i + g**2)

        h = npfft.fftshift(h)
        h /= h.sum()
        self._h = npfft.fft2(h)

    def _apply(self, x):

        if not self._conj:
            h = self._h
        else:
            h = self._h.conj()

        y = npfft.ifft2(
            h * npfft.fft2(x.reshape(self._in_signal_shape))).reshape((-1, 1))

        if np.isrealobj(x):
            y = np.real(y)

        return y


'''
class opWavelet(opBase):
    """Wavelet operator.
    
    Create an operator that applies a given wavelet transform to
    a 2D input signal.
    """
    
    def __init__(self, shape, name='haar', levels=None, undecimated=False):
        """
        Parameters
        ==========
        shape : (m, n)
            Shape of the 2D input signal.
        name : str, optional
            Name of wavelet. Supported values:
                * '**haar**' : Haar wavelet
                * '**db1**'-'**db20**' : Daubechie wavelets with different support size
                * '**sym1**'-'**sym20**' : Symlets wavelets with different support size
                * '**coif1**'-'**coif5**' : Coiflets wavelets with different support size
                * '**bior1.{1,3,5}**', '**bior2.{2,4,6,8}**', '**bior3.{1,3,5,7}**', '**bior4.4**', '**bior5.5**', '**bior6.8**': Biorthogonal wavelets with different support size
                * '**dmey**' : Discrete Meyer wavelet
        levels : integer, optional (default=None)
            Number of levels in the transformation. Both `m` and `n` must
            be divisible by 2**levels. If ``None`` then the number of levels
            is calculated automatically.
        undecimated : boolean, optional (default=False)
            Indicates wether to use the standard or the undecimated transform:
                * True - standard transform
                * False - undecimated
        """
        
        assert len(shape) == 2, "opWavelet supports operations on 2D matrices only"
        
        if levels == None:
            #
            # Calculate the maximal level according
            # to signal size
            #
            m, n = shape
            i = n
            j = 0
            while i % 2 == 0:
                i >>= 1
                j += 1
    
            levels = m
            i = 0
            while levels % 2 == 0:
                levels >>= 1
                i += 1
                
            if min(m, n) == 1:
                levels = max(i, j)
            else:
                levels = min(i, j)

        size_out = shape[0] * shape[1]
        if undecimated:
            shape_in = (shape[0], shape[1]*(1+3*levels))
        else:
            shape_in = shape
        size_in = shape_in[0] * shape_in[1]

        super(opWavelet, self).__init__(
            name='Wavelet',
            shape=(size_out, size_in),
            in_signal_shape=shape_in,
            out_signal_shape=shape
        )
        
        cl, ch, rl, rh = wavelets.waveletCoeffs(name)

        if undecimated:
            w = shape[1]
            def iwrap(x):
                x = x.reshape(self._in_signal_shape)
                xl = x[:, :w]
                xh = x[:, w:]
                y, l = rwt.irdwt(xl, xh, rl, rh, levels)
                return y
            
            def wrap(x):
                x = x.reshape(self._out_signal_shape)
                yl, yh, l = rwt.rdwt(x, cl, ch, levels)
                #
                # The order parameter is used to force a contigous
                # array.
                #
                return np.array(np.hstack((yl, yh)), order='C')
        else:
            def iwrap(x):
                x = x.reshape(self._in_signal_shape)
                y, l = rwt.idwt(x, rl, rh, levels)
                return y
            
            def wrap(x):
                x = x.reshape(self._out_signal_shape)
                y, l = rwt.dwt(x, cl, ch, levels)
                return y

        self._idwt = iwrap
        self._dwt = wrap
        
    def _apply(self, x):
        
        if self._conj:
            wf = self._dwt
        else:
            wf = self._idwt
            
        if np.isrealobj(x):
            y = wf(x)
        else:
            y1 = wf(x.real)
            y2 = wf(x.imag)
            y = y1 + 1j*y2
             
        y.shape = (-1, 1)
        return y
'''


class opFFT2d(opBase):
    """Two-dimensional fast Fourier transform (FFT) operator.

    Create an operator that applies a normalized fourier transform to
    a 2D input signal.
    """

    def __init__(self, shape):
        """
        Parameters
        ==========
        shape : (m, n)
            Shape of the 2D input signal.
        """

        assert len(shape) == 2, "opFFT2d supports operations on 2D matrices only"
        size = shape[0] * shape[1]

        super(opFFT2d, self).__init__(
            name='FFT2d',
            shape=(size, size),
            in_signal_shape=shape
        )

        self._normalization_coeff = np.sqrt(size)

    def _apply(self, x):

        if self._conj:
            y = npfft.ifft2(x.reshape(self._in_signal_shape)) * \
                self._normalization_coeff
        else:
            y = npfft.fft2(x.reshape(self._in_signal_shape)) / \
                self._normalization_coeff

        y = np.ascontiguousarray(y).reshape((-1, 1))
        y = np.real_if_close(y, tol=1e6)
        return y


class opDCT(opBase):
    """Arbitrary dimensional discrete cosine transform (DCT).

    Create an operator that applies the discrete cosine transform
    to vectors of arbitray dimension.
    """

    def __init__(self, shape, axis=-1):
        """
        Parameters
        ==========
        shape : list of integers
            Shape of the input signal.
        axis : integer, optional (default=-1)
            Axis along which the dct is computed. If -1 then the
            transform is multidimensional(default=-1)
        """

        _shape = [int(i) for i in shape]
        assert list(shape) == _shape, "shape must be a list of integers"
        assert axis > - \
            1 or axis < len(
                shape), "axis must be either -1 or one of the dimension indices"
        size = np.prod(shape)

        super(opDCT, self).__init__(
            name='DCT',
            shape=(size, size),
            in_signal_shape=shape
        )

        self._axis = axis

    def _apply(self, x):

        if self._conj:
            f = spfft.idct
        else:
            f = spfft.dct

        x = x.reshape(self._in_signal_shape)

        if self._axis == -1:
            y = x
            for i in range(x.ndim):
                y = f(y, axis=i, norm='ortho')
        else:
            y = f(x, axis=self._axis, norm='ortho')

        y.shape = (-1, 1)
        return y


class opDirac(opBase):
    """Identity operator

    Create an operator whose output signal equals the input signal.
    """

    def __init__(self, shape):
        """
        Parameters
        ==========
        shape : list of integers
            Shape of the input signal.
        """

        if isinstance(shape, (int, long)):
            shape = [shape]

        _shape = [int(i) for i in list(shape)]
        assert list(shape) == _shape, "shape must be a list of integers"
        size = np.prod(shape)

        super(opDirac, self).__init__(name='Dirac',
                                      shape=(size, size), in_signal_shape=shape)

    def _apply(self, x):

        return x.copy()


class opFoG(opBase):
    """Concatenate a sequence of operators into a single operator.
    """

    def __init__(self, operators_list):
        """
        Parameters
        ==========
        operators_list : list
            List of operators. All the operators must be instances of
            `opBase` or its subclasses. The `opFoG` operator applies
            the operators to the input signal in reverse order, i.e.
            starting with `operators_list[-1]`.
        """
        if len(operators_list) == 0:
            raise Exception('At least one operator must be specified')

        #
        # Check operator consistency and space
        #
        m, n = operators_list[0].shape

        for oper in operators_list[1:]:
            m_, n_ = oper.shape

            if m_ != n:
                raise Exception(
                    'Operator %s is not consistent with the previous operators' % oper.name)

            n = n_

        super(opFoG, self).__init__(
            name='FoG',
            shape=(m, n),
            in_signal_shape=operators_list[-1].in_signal_shape,
            out_signal_shape=operators_list[0].out_signal_shape
        )
        self._operators_list = operators_list

    @property
    def operators_list(self):
        """The list of operators that make up the opFoG.
        """
        if self._conj:
            return [op.T for op in self._operators_list[::-1]]
        else:
            return self._operators_list

    def _apply(self, x):

        if self._conj:
            y = self._operators_list[0].T(x)
            for oper in self._operators_list[1:]:
                y = oper.T(y)
        else:
            y = self._operators_list[-1](x)
            for oper in self._operators_list[-2::-1]:
                y = oper(y)

        return y


class op3DStack(opBase):
    """Extend an operator to process a stack of signals.

    The op3DStack operator is useful for example when the input signal
    is a stack of images and the base operator is applied to each
    image separately.
    """

    def __init__(self, operator, dim3):
        """
        Parameters
        ==========
        operator : instance of a subclass of opBase 
            The base operator. This operator is applied separately
            to each of the sections that make up the stacked input
            signal.
        dim3 : integer
            The size of the stack.
        """
        if not isinstance(operator, opBase):
            raise Exception('operator should be an instance of opBase.')

        #
        # Check operator consistency and space
        #
        m, n = operator.shape

        in_signal_shape = operator.in_signal_shape
        if in_signal_shape[1] == 1:
            in_signal_shape = (in_signal_shape[0]*dim3, 1)
        else:
            in_signal_shape = (in_signal_shape[0], in_signal_shape[1], dim3)

        out_signal_shape = operator.out_signal_shape
        if out_signal_shape[1] == 1:
            out_signal_shape = (out_signal_shape[0]*dim3, 1)
        else:
            out_signal_shape = (out_signal_shape[0], out_signal_shape[1], dim3)

        super(op3DStack, self).__init__(
            name='3DStack',
            shape=(m*dim3, n*dim3),
            in_signal_shape=in_signal_shape,
            out_signal_shape=out_signal_shape
        )
        self._operator = operator
        self._dim3 = dim3

    def _apply(self, x):

        if self._conj:
            op = self._operator.T
        else:
            op = self._operator

        y = []
        for x_ in np.split(x, self._dim3):
            y.append(op(x_))

        return np.vstack(y)


class opRandMask(opBase):
    """Random binary mask.

    The opRandMask operator creates and applies a random binary mask.
    """

    def __init__(self, shape, fill_ratio):
        """
        Parameters
        ==========
        shape : list of integers
            Shape of the input signal.
        fill_ratio : float
            Ratio of non zero (1) values in the mask.
        """

        _shape = [int(i) for i in shape]
        assert list(shape) == _shape, "shape must be a list of integers"
        assert fill_ratio > 0 and fill_ratio < 1, "fill_ratio must be a float in the range (0, 1)"
        size = np.prod(shape)

        super(opRandMask, self).__init__(
            name='RandomMask',
            shape=(size, size),
            in_signal_shape=shape
        )

        self._mask = np.zeros(shape, dtype=np.bool)
        indices = np.arange(size)
        np.random.shuffle(indices)
        indices = indices[:int(size*fill_ratio)]
        self._mask.ravel()[indices] = 1

    def _apply(self, x):

        x.shape = self.in_signal_shape
        y = x * self._mask
        y.shape = (-1, 1)
        return y


def test_DCT():
    """
    Test the opDCT operator
    """

    from scipy.misc import lena
    import matplotlib.pyplot as plt
    from compsense.utilities import softThreshold, hardThreshold

    img = lena().astype(np.double)
    img /= img.max()
    op = opDCT(img.shape)

    img_conv = op.T(img)
    img_recon = op(hardThreshold(img_conv, 0.5))

    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.figure()
    plt.imshow(img_conv)
    plt.figure()
    plt.imshow(img_recon)

    plt.show()


def test_FFT():
    """
    Test the opFFT2d operator
    """

    from scipy.misc import lena
    import matplotlib.pyplot as plt

    img = lena().astype(np.double)
    img /= img.max()
    op = opFFT2d(img.shape)

    img_conv = op(img)
    img_recon = op.T(img_conv)

    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.figure()
    plt.imshow(np.abs(img_conv))
    plt.figure()
    plt.imshow(img_recon)

    plt.show()


def test_RandomMask():
    """
    Test the opRandMask operator
    """

    from scipy.misc import lena
    import matplotlib.pyplot as plt

    img = lena().astype(np.double)
    img /= img.max()
    op = opRandMask(img.shape, fill_ratio=0.6)

    img_masked = op(img)

    plt.figure()
    plt.gray()
    plt.imshow(img_masked)

    plt.show()


if __name__ == '__main__':

    test_RandomMask()
