"""Defines a simple meta-learner
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from torch.autograd import Variable
import argparse

"""the one should have
At each timestep, it receives (s, a, r, d) as input, which is embedded and provided as input to the rnn

2 input units - for the action that it had, and the observation (reward) at each timestep... what if they're different scales?

n hidden units - this is the core of the algorithm. since it's complicated (just confidence bounds?) it can be small+simple

1 output unit - to output the next lever
"""


def focus(message):
    print('--' * 20)
    print(message)
    print('--' * 20)

if __name__ == '__main__':  # Avoid defining flags when used as a library.
    parser = argparse.ArgumentParser(
        description='Builds a simple RNN'
    )
    parser.add_argument(
        '--hidden_size', type=int, default=20,
        help='dimension of the hidden vector in the RNN'
    )
    parser.add_argument(
        '--layers', type=int, default=1,
        help='number of layers in the RNN'
    )
    parser.add_argument(
        '--debug', type=bool, default=False,
        help='whether to print debug messages'
    )
    FLAGS = parser.parse_args()


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=20, layers=1, input_size=3, output_size=2, timesteps=1):
        super(SimpleRNN, self).__init__()
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.layers = layers

        self.embed = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, layers, dropout=0.05)
        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        layers = self.layers
        n = self.hidden_size
        t = self.timesteps
        return (torch.randn(layers, t, n), torch.randn(layers, t, n))

    def forward(self, obs):
        # we first embed the observation
        prep = self.embed(obs)

        # then we feed it to the rnn
        out, self.hidden = self.rnn(prep, self.hidden)

        # finally, the output is passed through a fully connected layer, followed by a softmax
        actions = self.linear(out)
        action_softmax = F.log_softmax(actions, dim=2)

        if FLAGS.debug:
            focus("testing forward pass")
            print(prep)
            print(out)
            print(actions)
            print(action_softmax)

        return action_softmax
        

def generateSampleData(timesteps, batch_size):
    """Generate some dummy data
    
    the `obs` will have a shape of (batch_size, n_time, n_dimension)
    """
    episode = [[0.0] * timesteps] # timesteps x 1
    obs = torch.tensor([episode for i in range(batch_size)])
    return obs

def testSimpleRNN():
    obs = generateSampleData(timesteps=3,batch_size=1)
    a = SimpleRNN(FLAGS.hidden_size, FLAGS.layers, timesteps=1)
    a.forward(obs)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(a.parameters(), lr=0.1)

    if FLAGS.debug:
        focus("parameters")
        for name, param in a.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

if __name__ == "__main__":
    testSimpleRNN()