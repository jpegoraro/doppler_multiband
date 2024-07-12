import numpy as np
import scipy as sp


def find_peaks_mod(s, distance=3, height=0.0):
    # workaround to catch peaks in index 0
    s = np.concatenate(([min(s)], s, [min(s)]))
    peaks, vals = sp.signal.find_peaks(s, distance=distance, height=height)
    peaks -= 1  # -1 because we padded the pdp
    return peaks, vals
