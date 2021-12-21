import networkx as nx
import numpy as np
# import types
from networkx import Graph
from numpy import ndarray

# Base parameters
MORTALITY_RATE=0.012
GLOBAL_BIRTH_RATE=0.000185
GLOBAL_DEATH_RATE=0.0007645
# 2.5 people get infected for each infected person
# on average
INFECTION_RATE=2.5
INCUBATION_PERIOD=5

class SEIRModel:
    def __init__(self, initial_state: ndarray, population,
                 birth_rate: float=GLOBAL_BIRTH_RATE,
                 death_rate: float=GLOBAL_DEATH_RATE,
                 infection_rate: float=INFECTION_RATE,
                 mortality_rate: float=MORTALITY_RATE):
        self.initial_state = initial_state
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.infection_rate = infection_rate / population
        self.mortality_rate = mortality_rate
        self.current_state = initial_state
        self.population = population
        self.t = 0

    def step(self, nsteps: int=1):
        """Progress the model by `nsteps` time steps, where a single
        time step is a 24 hour period. Returns the resulting status.

        Parameters
        ----------
        nsteps: int
            The number of time steps to take.

        Returns
        -------
        current_state: NumPy array
            The number of people in each category (S, I, R, D).
        """
        for _ in range(nsteps):
            s = self.current_state[0]
            i = self.current_state[1]
            r = self.current_state[2]
            # The population doesn't have to add up to 1 in this case
            # because we are also accounting for natural births and
            # deaths. Perhaps we should use the actual number rather
            # than proportions?
            p = s+i+r

            self.current_state[0] += (self.birth_rate*p
                                   - self.infection_rate*s*i
                                   - self.death_rate*s)
            self.current_state[1] += (self.infection_rate*s*i
                                   - (1-self.mortality_rate-self.death_rate)*i
                                   - self.mortality_rate*i
                                   - self.death_rate*i)
            self.current_state[2] += (1 - self.mortality_rate
                                   - self.death_rate)*i
            self.current_state[3] += (self.mortality_rate*i
                                   + self.death_rate*p)

            self.t += 1

        return self.current_state * self.population
