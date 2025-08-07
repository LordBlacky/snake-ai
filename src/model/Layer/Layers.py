import numpy as np


class FullyConnected:

    def __init__(self, input_size: int, output_size: int, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.intermediate_value = None
        self.weight_tensor = None
        self.gradient_tensor = None

    def initialize(self, initializer):
        bias = initializer.init_bias((1, self.output_size))
        weights = initializer.init_weights((self.input_size, self.output_size))
        self.weight_tensor = np.vstack([weights, bias])

    def forward(self, input_tensor):
        ones = np.ones((input_tensor.shape[0], 1))
        input_tensor_extended = np.hstack([input_tensor, ones])
        self.intermediate_value = input_tensor_extended
        return self.activation_function.forward(
            input_tensor_extended @ self.weight_tensor
        )

    def backward(self, error_tensor):
        error_tensor = self.activation_function.backward(error_tensor)
        self.gradient_tensor = self.intermediate_value.T @ error_tensor
        return error_tensor @ self.weight_tensor[:-1].T

    def update(self, optimizer):
        self.weight_tensor = optimizer.calculate_update(
            self.weight_tensor, self.gradient_tensor
        )
