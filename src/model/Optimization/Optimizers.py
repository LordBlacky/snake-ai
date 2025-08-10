import numpy as np


class SGD:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, parameter_tensor, gradient_tensor):
        return parameter_tensor - self.learning_rate * gradient_tensor


class GeneticOptimizer:

    def __init__(self, mutation_rate: float, mutation_strength: float):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def calculate_update(self, parameter_tensor, gradient_tensor):
        mutation_mask = np.random.rand(*parameter_tensor.shape) < self.mutation_rate
        mutation_noise = np.random.normal(
            0, self.mutation_strength, size=parameter_tensor.shape
        )
        return np.clip(parameter_tensor + mutation_noise * mutation_mask, -1, 1)
