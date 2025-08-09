from model.Training.snake_evolutionary_genetic import SnakeAI

if __name__ == "__main__":
    ai = SnakeAI(50, 11)
    ai.init_population()
    ai.train()
