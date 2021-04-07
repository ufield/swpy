import numpy as np

def standardize(a, mean, var):
    return (a - mean)/np.sqrt(var)

def inverse_standardize(a, mean, var):
    return mean + np.sqrt(var)*a