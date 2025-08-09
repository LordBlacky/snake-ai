import numpy as np
import enum


class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Command(enum.Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    TURN_UP = 2
    TURN_DOWN = 3
    KEEP_DIRECTION = 4


class Game:
    def __init__(self, size):
        self.size = size
        self.direction = Direction.RIGHT
        self.body = [(size // 2, size // 2)]
        self.food = None
        self.lifetime = 0
        self.was_food_eaten_this_move = False
        self.game_over = False

    def is_head_out_of_bounds(self):
        x, y = self.body[0]
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        return False

    def has_self_collision(self):
        return self.body[0] in self.body[1:]

    def is_game_over(self):
        return self.game_over

    def check_conditions(self):
        if self.is_head_out_of_bounds() or self.has_self_collision():
            self.game_over = True
        else:
            self.game_over = False
        return self.game_over

    def spawn_food(self):
        all_coords = [(x, y) for x in range(self.size) for y in range(self.size)]
        free_coords = list(set(all_coords) - set(self.body))
        if free_coords:
            self.food = free_coords[np.random.randint(len(free_coords))]
        else:
            self.food = None
            self.game_over = True

    def reset(self):
        self.direction = Direction.RIGHT
        self.body = [(self.size // 2, self.size // 2)]
        self.game_over = False
        self.lifetime = 0
        self.was_food_eaten_this_move = False
        self.spawn_food()

    def evaluate_command(self, command):
        if command == Command.KEEP_DIRECTION:
            return
        if command == Command.TURN_LEFT:
            self.direction = Direction.LEFT
        elif command == Command.TURN_RIGHT:
            self.direction = Direction.RIGHT
        elif command == Command.TURN_UP:
            self.direction = Direction.UP
        elif command == Command.TURN_DOWN:
            self.direction = Direction.DOWN

    def move_and_check_food(self, command):
        self.evaluate_command(command)
        self.lifetime += 1
        head_x, head_y = self.body[0]
        if self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        self.body = [new_head] + self.body
        if new_head != self.food:
            self.body.pop()
            self.was_food_eaten_this_move = False
        else:
            self.was_food_eaten_this_move = True
            self.spawn_food()
        return self.check_conditions()

    def get_status(self):
        return self.game_over, len(self.body)

    def sample_command_from_distribution(self, probs):
        commands = [
            Command.TURN_LEFT,
            Command.TURN_RIGHT,
            Command.TURN_UP,
            Command.TURN_DOWN,  # ,
            # Command.KEEP_DIRECTION,
        ]
        probs = probs.flatten()
        idx = np.random.choice(4, p=probs)
        return commands[idx]

    def get_sensor_data(self):
        directions = [
            (0, -1),  # N
            (1, -1),  # NE
            (1, 0),  # E
            (1, 1),  # SE
            (0, 1),  # S
            (-1, 1),  # SW
            (-1, 0),  # W
            (-1, -1),  # NW
        ]

        head_x, head_y = self.body[0]
        sensor_data = []

        for dx, dy in directions:
            distance_to_wall = 0
            is_tail = 0
            is_food = 0

            x, y = head_x, head_y
            step = 1

            while 0 <= x < self.size and 0 <= y < self.size:
                x += dx
                y += dy
                step += 1

                if not (0 <= x < self.size and 0 <= y < self.size):
                    distance_to_wall = 1 / step
                    break

                if (x, y) in self.body[1:] and is_tail == 0:
                    is_tail = 1

                if self.food == (x, y) and is_food == 0:
                    is_food = 1

            sensor_data.extend([distance_to_wall, is_tail, is_food])

        return np.array([sensor_data])

    def print_board(self):
        board = [["." for _ in range(self.size + 1)] for _ in range(self.size + 1)]

        if self.food:
            fx, fy = self.food
            board[fy][fx] = "F"

        for i, (x, y) in enumerate(self.body):
            if i == 0:
                board[y][x] = "H"
            else:
                board[y][x] = "s"

        for row in board:
            print(" ".join(row))
