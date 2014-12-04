#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def heatmap(v):
    """
    Plot a grayscale heatmap given a vector of 
    state probabilities.
    """
    N = int(np.sqrt(len(v)))
    plt.imshow(v.reshape([N,N]),cmap = cm.Greys_r)
    return None
