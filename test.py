#тест среды и агента
import numpy as np
import os
from itertools import count
from torch.nn import Sequential, ReLU, Linear, BCELoss, Dropout, BatchNorm1d, Module
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from torch import device, cuda, Tensor, FloatTensor, save, load, cat, tensor, float
from torch.optim import Adam
from RNNSynthesis.helper import get_feature, get_feature_bits
from RNNSynthesis.environment import SimpleSynthesis
from CGRtools.files import SDFRead
from main import Actor, Critic, compute_returns, train_iters

os.environ['DB'] = './data/db.shelve'
os.environ['DATA'] = './data/rules/'

action_size = 1000
# device = device("cuda" if cuda.is_available() else "cpu")
target = next(SDFRead(open('/data/tylenol.sdf', encoding='UTF-8')))
env = SimpleSynthesis(target, steps=action_size)

# if __name__ == '__main__':
#     action_size = 1000
#     if os.path.exists('model/actor.pkl'):
#         actor = load('model/actor.pkl')
#         print('Actor Model loaded')
#     else:
#         actor = Actor(action_size).to(device)
#     if os.path.exists('model/critic.pkl'):
#         critic = load('model/critic.pkl')
#         print('Critic Model loaded')
#     else:
#         critic = Critic(action_size).to(device)
#     train_iters(actor, critic, n_iters=100)