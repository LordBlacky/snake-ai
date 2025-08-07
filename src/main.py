from model.Training.function_approximation import FunctionApproximator

if __name__ == '__main__':
    approximator = FunctionApproximator()
    approximator.train(100000)
