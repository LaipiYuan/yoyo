import os
import numpy as np
import math


def rle_decode(rle, shape):
    """
    rle: run-length string or list of pairs of (start, length)
    shape: (height, width) of array to return
    Returns
    -------
        np.array: 1 - mask, 0 - background
    """
    if isinstance(rle, float) and math.isnan(rle):
        rle = []
    if isinstance(rle, str):
        rle = [int(num) for num in rle.split(' ')]
    # [0::2] means skip 2 since 0 until the end - list[start:end:skip]
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    img = img.reshape(1, shape[0], shape[1])
    img = img.T
    return img


def rle_encode(img):
    """
    img: np.array: 1 - mask, 0 - background
    Returns
    -------
    run-length string of pairs of (start, length)
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle if rle else float('nan')


def RLenc(img, order='F'):
    """Convert binary mask image to run-length array or string.

    Args:
    img: image in shape [n, m]
    order: is down-then-right, i.e. Fortran(F)
    string: return in string or array

    Return:
    run-length as a string: <start[1s] length[1s] ... ...>
    """
    bytez = img.reshape(img.shape[0] * img.shape[1], order=order)
    bytez = np.concatenate([[0], bytez, [0]])
    runs = np.where(bytez[1:] != bytez[:-1])[0] + 1  # pos start at 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Use for sanity check the encode function
def RLdec(rl_string, shape=(101, 101), order='F'):
    """Convert run-length string to binary mask image.

    Args:
    rl_string:
    shape: target shape of array
    order: decode order is down-then-right, i.e. Fortran(F)

    Return:
    binary mask image as array
    """
    s = rl_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order=order)