"""Defines multi-arm bandit problem
"""
from random import random

class BanditProblem:
    """Defines a bandit problem, where the rewards are fixed

    n - number of arms
    probs - probabilities
    """

    def __init__(self, n=2):
        self.n = n
        self.probs = [random() for i in range(self.n)]
    
    
    def pull(self, arm):
        """Returns a random reward for pulling the arm
        """
        if random() <= self.probs[arm]:
            reward = 1
        else:
            reward = 0
        
        return reward