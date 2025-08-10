from model.Training.snake_evolutionary_genetic import SnakeAI
from snake.snake_game_headless import Game

if __name__ == "__main__":
    ai = SnakeAI(50, 6)
    ai.init_population()
    ai.pre_train_on_data(40)
    ai.train()
    #game = Game(40)
    #game.log_training_data()
