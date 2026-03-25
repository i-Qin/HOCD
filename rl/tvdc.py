import numpy as np


class TVDC:
    def __init__(self, config=None):
        self._alpha1 = 0.75
        self._alpha2 = 0.22
        self._u1 = 2
        self._u2 = 3
        self._u3 = 1
        self._u4 = 3
        self._min_lambda = 0.1

    def choose_action(self, state, train):
        e_d = state[1]
        e_fi = state[2]
        # Involvement:  concentrated-0.6  normal-：0.45  fatigue:0.3
        DI = 0.45
        # Ability
        DA = 1 / (1 + (self._alpha1*e_d)**2 + (self._alpha2*e_fi)**2)

        authority = np.exp(-((self._u1*DI)**self._u2) * ((self._u3*DA)**self._u4))
        authority = max(authority, 0.1)
        return authority



