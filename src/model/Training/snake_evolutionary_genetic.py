from model.Network.Networks import FeedForward
from model.Activation.ActivationFunctions import ReLU, Tanh, SoftMax, Sigmoid, Linear
from model.Initialization.Initializers import Normal
from model.Layer.Layers import FullyConnected
from model.Loss.LossFunctions import MSE
from model.Optimization.Optimizers import SGD, GeneticOptimizer
from snake.snake_game_headless import Game

import numpy as np
import copy
import math
import os


class Member:

    def __init__(self, network, game):
        self.network: FeedForward = network
        self.game: Game = game
        self.fitness = None

    def calculate_fitness(self):
        self.fitness = self.game.get_status()[1]


class SnakeAI:

    def __init__(self, epochs, population_exponent):
        self.epochs = epochs
        self.population_size = 2**population_exponent
        self.population: list[Member] = []

    def init_population(self):
        for _ in range(self.population_size):
            network = FeedForward(Normal(), GeneticOptimizer(0.1, 1), MSE())
            network.append_layer(FullyConnected(24, 18, ReLU()))
            network.append_layer(FullyConnected(18, 18, ReLU()))
            network.append_layer(FullyConnected(18, 5, SoftMax()))
            network.initialize()
            game = Game(50)
            game.reset()
            self.population.append(Member(network, game))

    def train(self):
        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            for member in self.population:
                while not member.game.is_game_over():
                    member.game.move_and_check_food(
                        member.game.sample_command_from_distribution(
                            member.network.test(member.game.get_sensor_data())
                        )
                    )
                    member.game.print_board()
                    os.system('clear')
                member.calculate_fitness()

            self.population.sort(key=lambda m: m.fitness, reverse=True)
            print("Best Fitness: ", self.population[0].fitness)
            num_survivors = max(1, math.ceil(self.population_size * 0.05))
            survivors = self.population[:num_survivors]
            new_population = []
            new_population.extend(survivors)
            clones_per_survivor = self.population_size // num_survivors

            for survivor in survivors:
                for _ in range(clones_per_survivor):
                    cloned_net = copy.deepcopy(survivor.network)
                    cloned_net.update()
                    game = Game(50)
                    game.reset()
                    new_population.append(Member(cloned_net, game))

            self.population = new_population[: self.population_size]
