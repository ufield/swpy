import numpy as np


def standardize(a, mean, var):
    return (a - mean)/np.sqrt(var)

def inverse_standardize(a, mean, var):
    return np.sqrt(var)*a + mean