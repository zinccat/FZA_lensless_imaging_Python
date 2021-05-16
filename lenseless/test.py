import numpy as np
import scipy.ndimage
import scipy.signal
import cv2

X = np.array([1, 2, 3]).reshape(1, 3)  # input data
H = np.array([4, 5, 6]).reshape(1, 3)  # kernel

# display scipy.signal.correlate2 results
print("scipy.signal.correlate2d results")
print("full: {}".format(scipy.signal.correlate2d(X, H, 'full')))
print("same: {}".format(scipy.signal.correlate2d(X, H, 'same')))
print("valid: {}".format(scipy.signal.correlate2d(X, H, 'valid')))

print('')
# display scipy.signal.convolve2 results
print("scipy.signal.convolve2d results")
print("full: {}".format(scipy.signal.convolve2d(X, H, 'full')))
print("same: {}".format(scipy.signal.convolve2d(X, H, 'same')))
print("valid: {}".format(scipy.signal.convolve2d(X, H, 'valid')))
