import numpy as np


class ReLU:

    def __init__(self):
        self.intermediate_value = None

    def forward(self, input_tensor):
        self.intermediate_value = input_tensor
        return np.maximum(input_tensor, 0)

    def backward(self, error_tensor):
        return error_tensor * (self.intermediate_value > 0)


class Linear:

    def __init__(self):
        self.intermediate_value = None

    def forward(self, input_tensor):
        self.intermediate_value = input_tensor
        return input_tensor

    def backward(self, error_tensor):
        return error_tensor


class Tanh:

    def __init__(self):
        self.intermediate_value = None

    def forward(self, input_tensor):
        self.intermediate_value = input_tensor
        return np.tanh(input_tensor)

    def backward(self, error_tensor):
        return error_tensor * (1 - np.tanh(self.intermediate_value) ** 2)


class Sigmoid:

    def __init__(self):
        self.intermediate_value = None

    def sigmoid(self, input_tensor):
        return 1 / (1 + np.exp(-input_tensor))

    def forward(self, input_tensor):
        self.intermediate_value = input_tensor
        return self.sigmoid(input_tensor)

    def backward(self, error_tensor):
        return (
            error_tensor
            * self.sigmoid(self.intermediate_value)
            * (1 - self.sigmoid(self.intermediate_value))
        )
