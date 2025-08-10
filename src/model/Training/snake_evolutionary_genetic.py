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
import time


class Member:

    def __init__(self, network, game):
        self.network: FeedForward = network
        self.game: Game = game
        self.fitness = None

    def calculate_fitness(self):
        score = self.game.get_status()[1]
        lifetime = self.game.lifetime

        if score < 10:
            self.fitness = math.floor(lifetime**2) * 2**score
        else:
            self.fitness = math.floor(lifetime**2) * 2**10 * (score - 9)


class SnakeAI:

    def __init__(self, epochs, population_exponent):
        self.epochs = epochs
        self.population_size = 2**population_exponent
        self.population: list[Member] = []
        self.fitness = [0]
        self.score = [0]

    def init_population(self):
        for _ in range(self.population_size):
            network = FeedForward(Normal(), GeneticOptimizer(0.05, 0.02), MSE())
            network.append_layer(FullyConnected(24, 18, ReLU()))
            network.append_layer(FullyConnected(18, 18, ReLU()))
            network.append_layer(FullyConnected(18, 4, ReLU()))
            network.initialize()
            game = Game(40)
            game.reset()
            self.population.append(Member(network, game))

    def train(self):
        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            mem_num = 0
            for member in self.population:
                mem_num += 1
                life_left = 200
                while not member.game.is_game_over() and life_left > 0:
                    member.game.move_and_check_food(
                        member.game.sample_command_from_distribution(
                            member.network.test(member.game.get_sensor_data())
                        )
                    )
                    life_left -= 1
                    if member.game.was_food_eaten_this_move:
                        if life_left < 500:
                            if life_left > 400:
                                life_left = 500
                            else:
                                life_left += 100
                    member.game.print_board()
                    print(
                        f"Epoch: {epoch}, life_left: {life_left}, BestFitness: {max(self.fitness)}, BestScore: {max(self.score)}, Member: {mem_num} / {self.population_size}"
                    )
                    time.sleep(0.05)
                    os.system("clear")
                member.calculate_fitness()

            self.population.sort(key=lambda m: m.game.get_status()[1], reverse=True)
            self.score.append(self.population[0].game.get_status()[1])
            self.population.sort(key=lambda m: m.fitness, reverse=True)
            self.fitness.append(self.population[0].fitness)
            num_survivors = max(1, math.ceil(self.population_size * 0.1))
            survivors = self.population[:num_survivors]
            new_population = []
            new_population.extend(survivors)
            clones_per_survivor = self.population_size // num_survivors

            for survivor in survivors:
                for _ in range(clones_per_survivor):
                    cloned_net = copy.deepcopy(survivor.network)
                    cloned_net.update()
                    game = Game(40)
                    game.reset()
                    new_population.append(Member(cloned_net, game))

            self.population = new_population[: self.population_size]
