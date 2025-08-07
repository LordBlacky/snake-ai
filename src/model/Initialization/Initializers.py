import numpy as np


class Normal:

    def __init__(self):
        pass

    @staticmethod
    def init_weights(shape):
        return np.random.standard_normal(shape)

    @staticmethod
    def init_bias(shape):
        return np.zeros(shape)
