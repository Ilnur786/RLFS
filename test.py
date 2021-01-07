#тест среды и агента

import os
os.environ['DB'] = './data/db.shelve'
os.environ['DATA'] = './data/rules/'

from torch import device, cuda, load
from RNNSynthesis.environment import SimpleSynthesis
from CGRtools.files import SDFRead
from main import Actor, Critic, train_iters

action_size = 1000
device = device("cuda" if cuda.is_available() else "cpu")
target = next(SDFRead(open('data/tylenol.sdf', encoding='UTF-8')))
env = SimpleSynthesis(target, steps=action_size)
if os.path.exists('model/actor.pkl'):
    actor = load('model/actor.pkl')
    print('Actor Model loaded')
else:
    actor = Actor(action_size).to(device)
if os.path.exists('model/critic.pkl'):
    critic = load('model/critic.pkl')
    print('Critic Model loaded')
else:
    critic = Critic(action_size).to(device)
train_iters(actor, critic, env, target, n_iters=1000)
