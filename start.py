
"""Generates a multi-arm bandit problem, and trains a network on it

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
import random
import numpy as np

from bandit import BanditProblem
from learner import SimpleRNN

if __name__ == '__main__':  # Avoid defining flags when used as a library.
    parser = argparse.ArgumentParser(
        description='Bandit problem for meta-learning agents'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=100,
        help='length of a trial (number of episodes before changing probabilities)'
    )
    parser.add_argument(
        '--debug', type=bool, default=True,
        help='whether to print debug messages'
    )
    FLAGS = parser.parse_args()

def main(args):
    l = []

    b = BanditProblem()

    print(b.probs)
    for t in range(100):
        reward = b.pull(random.randint(0,1))
        l.append(reward)
    print(l)
    print(np.mean(l))

if __name__ == '__main__':
    main(sys.argv)