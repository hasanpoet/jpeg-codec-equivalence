"""Entropy of quantized luma DCT coefficients at the reference JPEG table."""
import numpy as np
from scipy.fft import dctn
from .codecs import luma

Q50 = np.array([[16,11,10,16,24,40,51,61], [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56], [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77], [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101], [72,92,95,98,112,100,103,99]], dtype=float)


def complexity(pixels):
    blocks = (luma(pixels) - 128).reshape(32, 8, 32, 8).transpose(0, 2, 1, 3)
    coefficients = np.rint(dctn(blocks, axes=(-2, -1), norm='ortho') / Q50)
    counts = np.unique(coefficients, return_counts=True)[1]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))
