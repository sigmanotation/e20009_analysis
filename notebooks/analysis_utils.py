import math
from scipy.signal import fftconvolve
import numpy as np


def breit_wigner_shape(points: np.ndarray, Ei: float, gamma: float):
    """
    Unnormalized Breit-Wigner distribution that characterizes the shape of the spread
    of energies of a quantum mechanical state. It is calculated for each point in the
    input array for a state of input energy Ei and width gamma.

    Parameters
    ----------
    Ei: float
        Energy of the state in MeV.
    gamma: float
        Width of state in MeV.

    Returns
    -------
    np.ndarray
    """
    y = np.array([1 / ((point - Ei) ** 2 + gamma**2 / 4) for point in points])

    return y


def exp_line(points, Ei, gamma, a):
    bw = breit_wigner_shape(points, Ei, gamma)
    dr = detector_response(points)
    y = fftconvolve(bw, dr, mode="full", axes=0)

    return a * y[: len(points)]


# Detector response must be centered at 0 because it is a response! This might mean
# we need to lower the lower limit of the histograms more so that we capture more of it


def detector_response(points):
    """ """
    y = []
    for point in points:
        value: float = math.exp(-(point**2) / 0.2)
        y.append(value)

    return y


# this should probably be deleted. keeping for now
def shift_to_center(array: np.ndarray):
    """ """
    center = (array[-1] + array[0]) / 2
    array = array - center

    return array
