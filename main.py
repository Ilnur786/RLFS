import numpy as np
from torch.nn import Sequential, ReLU, Linear, BCELoss, Dropout, BatchNorm1d, Module
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from RNNSynthesis.helper import get_feature, get_feature_bits


# class Actor(Module):
#
#     def __init__(self,  learning_rate, activation=activation):
#         super(Actor, self).__init__()
#         self.lr = learning_rate
#         self.af = activation
#         self.
#
#     def forward(self, mc, state, action_space): #шаг вперед
#         fp = get_feature_bits(mc)
#
#
#     def loss(self): #на основании награды считать лосс функцию
#         ...

class Actor(Module):
    def __init__(self, action_size, inp_size, learning_rate):  # state == molecule_container
        super().__init__()
        self.inps = inp_size
        self.ac = action_size
        self.lr = learning_rate
        self.linear1 = Linear(self.inps, 128)
        self.linear2 = Linear(128, 256)
        self.linear3 = Linear(256, self.ac)

    def forward(self, state, activation=ReLU):
        state = get_feature_bits(state)
        output = activation(self.linear1(state))
        output = activation(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(softmax(output, dim=-1))
        return distribution


class Critic(Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = Linear(self.state_size, 128)
        self.linear2 = Linear(128, 256)
        self.linear3 = Linear(256, 1)

    def forward(self, state, activation=ReLU):
        state = get_feature_bits(state)
        output = activation(self.linear1(state))
        output = activation(self.linear2(output))
        value = self.linear3(output)
        return value


