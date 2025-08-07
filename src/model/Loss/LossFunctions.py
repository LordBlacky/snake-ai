import numpy as np


class MSE:

    def __init__(self):
        self.intermediate_value = None

    def forward(self, prediction_tensor, target_tensor):
        self.intermediate_value = prediction_tensor
        return np.mean((prediction_tensor - target_tensor) ** 2)

    def backward(self, target_tensor):
        B = target_tensor.shape[0]
        n = target_tensor.shape[1]
        return (2 / (B * n)) * (self.intermediate_value - target_tensor)
