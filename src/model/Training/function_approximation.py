from model.Network.Networks import FeedForward
from model.Activation.ActivationFunctions import ReLU, Linear, Tanh
from model.Initialization.Initializers import Normal
from model.Layer.Layers import FullyConnected
from model.Loss.LossFunctions import MSE
from model.Optimization.Optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt


class FunctionApproximator:

    def __init__(self):
        self.network = FeedForward(Normal(), SGD(0.01), MSE())

        self.network.append_layer(FullyConnected(1, 20, Tanh()))
        self.network.append_layer(FullyConnected(20, 1, Linear()))

        self.network.initialize()

    def train(self, epochs: int):
        x_values = np.arange(-10, 10, 0.01).reshape(-1, 1)
        y_values = x_values**2

        for _ in range(epochs):
            self.network.forward(x_values, y_values)
            self.network.backward()
            self.network.update()

        plt.scatter(range(len(self.network.loss)), self.network.loss)
        plt.show()

        predicted_y = self.network.test(x_values)
        plt.scatter(x_values, predicted_y)
        plt.show()
