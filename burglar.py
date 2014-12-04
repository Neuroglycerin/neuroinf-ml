#!/usr/bin/python
##########################################
# Script to run Barber's burglar example #
##########################################

import neuroml.HMM
import numpy as np

def main():
    """
    Runs the burglar example, return results.
    """

    # load the data
    creaks, bumps, observations, A = neuroml.HMM.load_burglar()

    # Create emission matrix
    B = np.vstack([creaks,bumps]).T

    # initialise the HMM
    HMM = neuroml.HMM.HMM(A,B)

    # loop over observations, filtering and saving results
    filter_results = {}
    for i,o in enumerate(observations):
        ht_dist = HMM.filter(o)
        filter_results[i] = ht_dist

    # plotting pending

    return filter_results


if __name__ == "__main__":
    main()
