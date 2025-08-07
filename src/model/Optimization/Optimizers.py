import numpy as np


class SGD:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, parameter_tensor, gradient_tensor):
        return parameter_tensor - self.learning_rate * gradient_tensor
