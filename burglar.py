#!/usr/bin/python
##########################################
# Script to run Barber's burglar example #
##########################################

import neuroml

def main():
    """
    Runs the burglar example, generates plots.
    """

    # load the data
    creaks, bumps, observations, A = neuroml.HMM.load_burglar()

    # REMEMBER TO TRANSPOSE EMISSION MATRIX

    return None


if __name__ == "__main__":
    main()
