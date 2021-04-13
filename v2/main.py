import math
import random

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from torch.nn import Sequential, ReLU, Linear, BCELoss, Dropout, BatchNorm1d, Module
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from torch import device, cuda, Tensor, FloatTensor, save, load, cat, tensor, float, unsqueeze
from torch.optim import Adam
import numpy as np
import os

os.environ['DB'] = os.path.abspath('../data/db.shelve')
os.environ['DATA'] = os.path.abspath('../data/rules')

from RNNSynthesis.helper import get_feature, get_feature_bits
from RNNSynthesis.environment import SimpleSynthesis
from CGRtools.files import SDFRead
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, num_outputs, hidden_size, num_inputs=8192 * 2, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)  # софт-макс возвращает распредление вероятностей между действиями, \
        return dist, value  # а категорикал возвращает случайное действие с вероятностью согласно распределию софт-макс
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


num_outputs = 10 #1000
# Hyper params:
hidden_size = 256
lr = 0.001
episode_steps = 5  # количество шагов в одном эпизоде. в конце такого эпизода мы вычисляем градинет и обновляем веса.

model = ActorCritic(num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

max_frames = 10000
frame_idx = 0
test_rewards = []

target = next(SDFRead(open('../data/target3.sdf', encoding='UTF-8')))
env = SimpleSynthesis(target, steps=max_frames)
env.action_space = [14586, 247896, 212011, 345698, 345655, 4234, 186945, 12689, 73862, 96521]
# истинный синтетический путь для target3 [212011, (345655,), (186945,), (73862,)], если в скобках указаны несколько реагентов, значит они взаимозаменяемы и нужно брать только один.

target_bits = get_feature_bits(target)
zero_arr = np.zeros_like(target_bits)

start_state = np.concatenate((zero_arr, target_bits), axis=None)
start_state = unsqueeze(Tensor(start_state), 0)
state = start_state
all_rewards = dict()
while frame_idx < max_frames:
    env.reset()
    log_probs = []
    values = [] #
    rewards = [] #копиться целый эпизод как награда от среды, потом используется (через специальную формулу)для расчета награды за целый эпизод игры и через спек
    masks = [] #ЗАЧЕМ МАСКА? для выяснения done или нет
    entropy = 0 #ЗАЧЕМ ЭНТРОПИЯ?

    for i in range(episode_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        #state, reward, done, info = env.step(action.item() + 1) для всего экшен спейса. + 1 т.к. у нее в среде начинается с 1, а не с 0
        state, reward, done, info = env.step(env.action_space[action.item()])
        if state:
            next_state = get_feature_bits(state)
            # last_state = next_state           # возможно зацикливается по причине того что стейт нан, и каждый раз вызывается этот стейт который вызывает нан
            state_plus_target = np.concatenate((next_state, target_bits), axis=None)
            next_state = unsqueeze(Tensor(state_plus_target), 0)

            log_prob = dist.log_prob(action)  #как вычисляется?
            # try:
            #     log_prob = dist.log_prob(action).unsqueeze(0)
            # except RuntimeError:
            #     print(action)
            #     return
            entropy += dist.entropy().mean()  # mean - среднее значение. ЗАЧЕМ ЭНТРОПИЯ?

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(tensor([reward], dtype=float, device=device))
            masks.append(tensor([1 - done], dtype=float, device=device))  # 1 - False = 1

            state = next_state
            if len(state) - len(target) >= 20:
                break
            if done:
                print(action, state, reward, done, info, env.steps, env.max_steps)
                if env.depth < 5 and not env.stop and env.steps <= env.max_steps:
                    print(action, state, reward, done, info)
                    print(f'синтетический путь для молекулы {target} = {[step for step in env.render()]}')
                    break
                else:
                    print('done, но молекула не синтезирована')
        else:
            reward = 0

            next_state = np.zeros_like(target)  # if last_state is None else last_state

            state_plus_target = np.concatenate((next_state, target), axis=None)
            next_state = unsqueeze(Tensor(state_plus_target), 0)
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()  # mean - среднее значение

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(tensor([reward], dtype=float, device=device))
            masks.append(tensor([1 - done], dtype=float, device=device))  # 1 - False = 1

            state = next_state
        print(f'reward per i{i} is {reward}')

        frame_idx += 1
    all_rewards[frame_idx] = max(rewards)
    print(frame_idx)
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    #ВОПРОС: КАК ПОНЯТЬ КОГДА НЕ НУЖНО ВЫЧИСЛЯТЬ ГРАДИНЕТ, Т.Е. ДЛЯ КАКИХ ПЕРМЕННЫХ (ТЕНЗОРОВ) НЕ НУЖНО ВЫЧИСЛЯТЬ ГРАДИЕНТ
    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()  # детач отключает от общего градиета,
    # т.е. для этой переменной градинет вичисляться не будет, видимо по причине того, что это
    values = torch.cat(values)

    advantage = returns - values  #ВЫГОДА - ...

    # Почему он (хозяин репозитория) так считает потери? сам придумал?
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    #

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#ОТРИСОВКА ГРАФИКОВ
filename = 'target3'
def reward_to_tanimoto(reward):
    return (1 + 10 ** (-10)) - (10 ** (0 - reward))   # todo: надо прервать все расчеты, перенести на сервере в нужные папки

#график с reward
df_reward = pd.DataFrame.from_dict(all_rewards, orient='index', columns=['ga_score'])

fig, ax = plt.subplots(figsize=(15, 7))
plt.title(f'Reward for {filename}')
plt.plot(df_reward)
plt.xlabel('steps')
plt.ylabel('ga_score')
plt.savefig(f'graphics/reward_{filename}.png')
# plt.show()

# график с танимомто
result_tanimoto = {key: reward_to_tanimoto(value) for key, value in all_rewards.items()}

df_tanimoto = pd.DataFrame.from_dict(result_tanimoto, orient='index', columns=['ga_score'])

fig1, ax1 = plt.subplots(figsize=(15, 7))
plt.title(f'Tanimoto for {filename}')
plt.plot(df_tanimoto)
plt.xlabel('steps')
plt.ylabel('Tanimoto')
plt.savefig(f'graphics/tanimoto_{filename}.png')
# plt.show()


fig2, ax2 = plt.subplots(figsize=(20,10))
plt.title(f'Reward and Tanimoto for {filename}')
ax2.plot(df_reward, sns.xkcd_rgb["denim blue"], alpha=1)
ax2.plot(df_tanimoto, sns.xkcd_rgb["pale red"], alpha=0.6)
ax2.legend(labels = ['Награда от среды','индекс Танимото'])
plt.savefig(f'graphics/all_{filename}.png')