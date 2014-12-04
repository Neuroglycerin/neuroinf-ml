#!/usr/bin/python

import numpy as np

def transition_probability(state_from, state_to, N=5):
    """
    Given a coordinate, return 
    the transition probabilities.
    Used to generate transition matrix.
    """
    
    prob = 0.0
    # Check if adjacent
    if abs(state_from[0] - state_to[0]) + abs(state_from[1] - state_to[1]) == 1:
        # Check for x edge
        x_edge = False
        y_edge = False
        if state_from[0] == 0 or state_from[0] == N-1:
            x_edge = True
        #Check for y edge
        if state_from[1] == 0 or state_from[1] == N-1:
            y_edge = True

        if x_edge and y_edge:
            prob = 1.0 / 2
        elif x_edge or y_edge:
            prob = 1.0 / 3
        else:
            prob = 1.0 / 4
    
    return prob

def load_burglar():
    """
    Function to load the necessary data structures
    for the burglar example. Such a small example
    hardcoding in the values isn't a problem.
    Returns:
        * creaks
        * bumps
        * observations
        * transition matrix
    """
    creaks = np.array([[0.1, 0.1, 0.1, 0.9, 0.9],
                       [0.1, 0.9, 0.9, 0.1, 0.1],
                       [0.1, 0.9, 0.1, 0.1, 0.1],
                       [0.9, 0.1, 0.1, 0.9, 0.9],
                       [0.9, 0.9, 0.1, 0.1, 0.1]])
    # this must be flattened to meet notation
    # in Barber
    creaks = np.ndarray.flatten(creaks)


    bumps = np.array([[0.9, 0.1, 0.1, 0.9, 0.9],
                      [0.9, 0.1, 0.1, 0.9, 0.9],
                      [0.1, 0.9, 0.1, 0.9, 0.1],
                      [0.1, 0.9, 0.1, 0.1, 0.1],
                      [0.9, 0.1, 0.1, 0.1, 0.1]])
    bumps = np.ndarray.flatten(bumps)

    observations = np.array([[1, 0],
                             [0, 1],
                             [1, 1],
                             [0, 0],
                             [1, 1],
                             [0, 0],
                             [0, 1],
                             [0, 1],
                             [0, 0],
                             [0, 0]])
    
    #
    N=5
    A = np.zeros([25,25])

    # iterate over all possible transitions in the matrix
    # for each calculate the probability that the transition could occur.
    for i in range(0,N**2):
        for j in range(0,N**2):
            # Svet's code, takes some thinking about but does work
            A[i,j] = transition_probability([i %N, i / N], [j % N, j / N])

    return creaks, bumps, observations, A

# define the HMM
class HMM():
    def __init__(self, transition, emission):
        """
        Initialise with transition probabilities
        and emission probabilities.
        Input:
            * Matrix of transition
            * Matrix of emission
        Output: None
        """
        # store initialisation
        self.transition = transition
        self.emission = emission

        # N states (DIFFERS FROM NOTATION IN NOTEBOOK)
        N = self.transition.shape[0]
        # initialise hidden state with uniform
        self.ht_dist = np.ones([N,1]) * 1.0 / N
        return None

    def _find_emission(self,observation):
        """
        Given an observation, find emission probabilities
        using emission distribution.
        """
        # find emission probabilities from emission matrix
        # iterating over observations and calculating emission
        # probabilities
        # python notes: [:] copies array
        emission_probabilities = self.emission[:]
        for i,o in enumerate(observation):
            if o == 0:
                # probability should be 1-emission column
                emission_probabilities[:,i] = 1-self.emission[:,i]
            # otherwise copy is correct
        # combine emission probabilities and convert to column vector 
        emission_probabilities = np.prod(emission_probabilities,
                                            axis=1)[np.newaxis].T
        
        return emission_probabilities

    def _alpha_update(self,alpha,observation):
        """Defined as joint distribution of h_t and v_1:t.
        Return a discrete probability distribution."""
        # find emission probabilities for this observation
        emission_probabilities = self._find_emission(observation)
        # calculate new alpha
        alpha = emission_probabilities * np.dot(self.transition,alpha)
        return alpha
    
    def filter(self, observation):
        """
        Take an observation and infer the present 
        hidden state, given transition and emission
        matrices.
        Input:
            * observation
        Output:
            * hidden state
        """
        try:
            self.alpha = self._alpha_update(self.alpha,observation)
        except AttributeError:
            # if we haven't made an alpha yet
            emission_probabilities = self._find_emission(observation)
            self.alpha = emission_probabilities * self.ht_dist

        # calculate new hidden state
        self.ht_dist = self.alpha/sum(self.alpha)

        return self.ht_dist
