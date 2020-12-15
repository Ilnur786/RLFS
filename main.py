import numpy as np
from torch.nn import Sequential, ReLU, Linear, Softmax, BCELoss, Dropout, BatchNorm1d, Module


class Actor(...):

    def __init__(self, learning_rate, activation=ReLU):
        self.lr = learning_rate
        self.act = activation

    def forward(self): #шаг вперед
        ...

    def loss(self): #на основании награды считать лосс функцию
        ...


