
"""Generates a multi-arm bandit problem, and trains a network on it

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import sys
import random
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical

from bandit import BanditProblem
from learner import SimpleRNN, makeObservation

if __name__ == '__main__':  # Avoid defining flags when used as a library.
    parser = argparse.ArgumentParser(
        description='Bandit problem for meta-learning agents'
    )
    parser.add_argument(
        '--n_episodes', type=int, default=20000,
        help='number of episodes (bandit problems) to train for'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=100,
        help='length of a trial (number of episodes before changing probabilities)'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.9,
        help='discount rate for the RL algorithm'
    )
    parser.add_argument(
        '--display_epochs', type=int, default=20,
        help='number of epochs to print'
    )
    parser.add_argument(
        '--debug', type=bool, default=True,
        help='whether to print debug messages'
    )
    FLAGS = parser.parse_args()


def assure_equal_and_not_empty(a,b):
    return len(a) == len(b) and len(a) > 0


def step(optimizer, records):
    """implements REINFORCE for learning
    """
    rewards = records.get("reward")
    log_probs = records.get("log_prob")
    assure_equal_and_not_empty(rewards, log_probs)

    R = 0
    discount_rewards = []
    policy_loss = []

    for r in rewards[::-1]:
        R = r + FLAGS.gamma * R
        discount_rewards.insert(0, R)

    for log_prob, reward in zip(log_probs, discount_rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    optimizer.step()

    
class EpisodeRecorder:
    def __init__(self):
        self.data = {}

    def record(self, thing, x):
        if thing in self.data:
            self.data[thing].append(x)
        else:
            self.data[thing] = [x]
    
    def get(self, thing):
        """Returns the recorded data
        
        Note: Raise an exception if `thing` was not recorded..
        """
        return self.data[thing]

       
def main(args):
    
    # create a meta-learner
    s = SimpleRNN(hidden_size=20, layers=2)
    optimizer = optim.SGD(s.parameters(), lr = 0.01, momentum=0.9)
    fantastic = EpisodeRecorder()

    K = 1
    for i in range(FLAGS.n_episodes // K):

        # reset the hidden state after every K environments
        s.reset_hidden()
        for y in range(K):
            start_time = time.time()
            
            # reset the environment
            action = 0
            reward = 0
            epi = EpisodeRecorder()
            b = BanditProblem()

            for t in range(FLAGS.sequence_length):
                done = (t == FLAGS.sequence_length - 1)

                # consult our neural network
                obs = makeObservation(0, action, reward, done)
                probs = s.forward(obs)
                epi.record("log_prob", probs)

                # choose the max action
                # (this is the part that isn't differentiable!)
                di = Categorical(probs)
                action = di.sample()
                epi.record("action", action)

                # update the last reward
                reward = b.pull(action)
                epi.record("reward", reward)

            # upgrade our neural network
            step(optimizer, epi)

            # before we destroy the environment
            fantastic.record("action_mean", np.mean(epi.get("action")))
            fantastic.record("action_var", np.std(epi.get("action")))
            fantastic.record("average_reward", np.mean(epi.get("reward")))
            fantastic.record("time", time.time() - start_time)

            # is it learning?
            current_epoch = K*i +y
            if current_epoch % FLAGS.display_epochs == 0:
                display = ""
                display += "Episode {}, Time (elapsed {:.0f}, {:.4f}s/episode), ".format(current_epoch, np.sum(fantastic.get("time")), np.mean(fantastic.get("time")[-FLAGS.display_epochs:]))
                display += "Reward (avg {:.4f}) ".format(np.mean(fantastic.get("average_reward")[-FLAGS.display_epochs:]))
                display += "Actions (std.dev {:.4f}) ".format(fantastic.get("action_var")[-1])
                print(display)

if __name__ == '__main__':
    main(sys.argv)
