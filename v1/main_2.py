import numpy as np
import os

os.environ['DB'] = '/home/ilnur/PycharmProjects/RLFS/data/db.shelve'
os.environ['DATA'] = '/home/ilnur/PycharmProjects/RLFS/data/rules'

from itertools import count
from torch.nn import Sequential, ReLU, Linear, BCELoss, Dropout, BatchNorm1d, Module
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from torch import device, cuda, Tensor, FloatTensor, save, load, cat, tensor, float, unsqueeze
from torch.optim import Adam
from RNNSynthesis.helper import get_feature, get_feature_bits
from RNNSynthesis.environment import SimpleSynthesis
from CGRtools.files import SDFRead
import random

device = device("cpu")

class Actor(Module):
    def __init__(self, action_size, inp_size=8192*2, learning_rate=0.0001, activation=ReLU):  # state == molecule_container
        super(Actor, self).__init__()
        self.inps = inp_size
        self.ac = action_size
        self.lr = learning_rate
        self.af = activation()
        self.linear1 = Linear(self.inps, 128)
        self.linear2 = Linear(128, 256)
        self.linear3 = Linear(256, self.ac)

    def forward(self, state):
        Tensor([1]).to(state.device)
        output = self.af(self.linear1(state))
        output = self.af(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(softmax(output, dim=-1))
        return distribution


class Critic(Module):
    def __init__(self, action_size, inp_size=8192*2, learning_rate=0.0001, activation=ReLU):
        super(Critic, self).__init__()
        self.inps = inp_size
        self.ac = action_size
        self.lr = learning_rate
        self.af = activation()
        self.linear1 = Linear(self.inps, 128)
        self.linear2 = Linear(128, 256)
        self.linear3 = Linear(256, 1)

    def forward(self, state):
        output = self.af(self.linear1(state))
        output = self.af(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train_iters(actor, critic, enviroment, target, n_iters):
    optimizer_a = Adam(actor.parameters())
    optimizer_c = Adam(critic.parameters())
    target_bits = get_feature_bits(target)
    zero_arr = np.zeros_like(target_bits)
    for iter in range(1, n_iters+1):
        start_state = np.concatenate((zero_arr, target_bits), axis=None)
        start_state = unsqueeze(Tensor(start_state), 0)
        state = start_state
        enviroment.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        last_state = None
        dist, value = actor(state), critic(state)
        #
        for i in count():  # счетчик количества шагов , за которое добрались до цели
            if i > 0:
                dist, value = actor(state), critic(state)  # dist - distribution (распределение вероятностей)

            action = dist.sample()
            state, reward, done, info = enviroment.step(action.item() + 1)
            molecule_container_state = state
            if state:
                next_state = get_feature_bits(state)
                #last_state = next_state           # возможно зацикливается по причине того что стейт нан, и каждый раз вызывается этот стейт который вызывает нан
                state_plus_target = np.concatenate((next_state, target_bits), axis=None)
                next_state = unsqueeze(Tensor(state_plus_target), 0)

                log_prob = dist.log_prob(action).unsqueeze(0)
                # try:
                #     log_prob = dist.log_prob(action).unsqueeze(0)
                # except RuntimeError:
                #     print(action)
                #     return
                entropy += dist.entropy().mean()  # mean - среднее значение

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(tensor([reward], dtype=float, device=device))
                masks.append(tensor([1 - done], dtype=float, device=device))  # 1 - False = 1

                state = next_state
                if len(molecule_container_state) - len(target) >= 20:
                    break
                if done:
                    if enviroment.depth < 5 and not enviroment.stop:
                        print(f'синтетический путь для молекулы {target} = {[step for step in enviroment.render()]}')
                        return
                    else:
                        print('done, но молекула не синтезирована')
            else:
                reward = 0

                next_state = np.zeros_like(target) #if last_state is None else last_state

                state_plus_target = np.concatenate((next_state, target), axis=None)
                next_state = unsqueeze(Tensor(state_plus_target), 0)
                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()  # mean - среднее значение

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(tensor([reward], dtype=float, device=device))
                masks.append(tensor([1 - done], dtype=float, device=device))  # 1 - False = 1

                state = next_state
            print(f'reward per i{i} is {reward}')
        #
        print(f'max reward per iter{iter} is {max(rewards)} !!!!!!!!!!!!!!!!!!!!!!!!')
        next_state = FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = cat(log_probs)  # конкатенация тензоров (по оси х)
        returns = cat(returns).detach()
        values = cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizer_a.zero_grad()
        optimizer_c.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizer_a.step()
        optimizer_c.step()
    save(actor, 'data/model/actor.pkl')
    save(critic, 'data/model/critic.pkl')
