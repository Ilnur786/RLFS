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
import random

os.environ['DB'] = './data/db.shelve'
os.environ['DATA'] = './data/rules'

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

# mine | gym:
# state = observation_space
# action_space = action_space
# reward = reward

# with SDFRead('/home/ilnur/Загрузки/molSet_largestCluster.sdf', 'rb') as f:
#     mc = next(f)
#
device = device("cuda" if cuda.is_available() else "cpu")
# lr = 0.0001
# env = SimpleSynthesis(mc)


class Actor(Module):
    def __init__(self, action_size, inp_size=8192, learning_rate=0.0001, activation=ReLU):  # state == molecule_container
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
    def __init__(self, action_size, inp_size=8192, learning_rate=0.0001, activation=ReLU):
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
    for iter in range(n_iters):
        enviroment.reset()
        r_action = random.choice(enviroment.action_space) # при r_action = 396925 , state получается None. После блока трай эксепт выполнение кода продолжается.
        state, reward, done, info = enviroment.step(r_action)
        try:
            state = get_feature_bits(state)
        except Exception:
            print(f'r_action={r_action}; state={state}; state type={type(state)} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(type(state))
        # state = np.array(state)  #моя вставка
        log_probs = []  # список ретурнов от лосс-функции log_prob() - список тензоров
        values = []
        rewards = []
        masks = []
        entropy = 0
        # enviroment.reset()

        for i in count():  # счетчик количества шагов , за которое добрались до цели
            enviroment.render()
            state = FloatTensor(state).to(device)  # to("CUDA"/"CPU") - переводит данные в память видеокарты / озу
            print(type(state))
            dist, value = actor(state), critic(state)  # dist - distribution (распределение вероятностей)

            action = dist.sample()
            try:
                next_state, reward, done, _ = enviroment.step(int(action.cpu().numpy())) #File "/home/ilnur/PycharmProjects/RL/venv/lib/python3.8/site-packages/RNNSynthesis/environment.py", line 85, in step
                                                                                            # assert action in self.action_space, \
                                                                                        # AssertionError: 0 (<class 'int'>) invalid
            except AssertionError:
                print(f'ошибка выпала на action={action}, ввиде int={int(action.cpu().numpy())}, тип последнего:{type(int(action.cpu().numpy()))} ??????????????????????????????????')

            next_state = get_feature_bits(next_state)

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()  # mean - среднее значение

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(tensor([reward], dtype=float, device=device))
            masks.append(tensor([1 - done], dtype=float, device=device))  # 1 - False = 1

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

            elif len(state) - len(target) >= 20:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

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
    save(actor, 'RL/data/model/actor.pkl')
    save(critic, 'RL/data/model/critic.pkl')
    # enviroment.close()


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
