import numpy as np

def strong_error_mc(x, xlim):
    return(np.mean(np.abs(x-xlim)))



