import random
import pygame
import numpy as np

class SnakeGame:
    def __init__(self, width=640, height=480, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        self.fps = 10  # Speed of the game when human is playing
        self.reset()

        self.pygame_initialized = False
        self.screen = None

    def init_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('AlphaSerpent')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 25)
            self.pygame_initialized = True

    def reset(self):
        """Reset the game to initial state, return the initial state"""
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2

        # Start with a snake of length 3
        self.snake = [(center_x * self.grid_size, center_y * self.grid_size),
                      ((center_x - 1) * self.grid_size, center_y * self.grid_size),
                      ((center_x - 2) * self.grid_size, center_y * self.grid_size)]

        # Start moving right
        self.direction = (self.grid_size, 0)

        # Place food
        self.food = self._place_food()

        # Initialize game variables
        self.score = 0
        self.game_over = False
        self.steps_without_food = 0
        self.max_steps_without_food = 100  # Prevent infinite loops

        # Return initial state
        return self._get_state()

    def _place_food(self):
        """Place food in a random location not occupied by snake"""
        while True:
            x = random.randint(0, self.grid_width - 1) * self.grid_size
            y = random.randint(0, self.grid_height - 1) * self.grid_size

            # Check if position is not part of snake
            if (x, y) not in self.snake:
                return (x, y)

    def _get_state(self):
        """
        Convert game state to a format suitable for the AI
        Returns a numpy array representation of the game state
        """
        # Create an empty grid
        state = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.float32)

        # Mark snake body positions with 1 in channel 0
        for segment in self.snake:
            x, y = segment
            state[y // self.grid_size, x // self.grid_size, 0] = 1

        # Mark snake head position with 1 in channel 1
        head_x, head_y = self.snake[0]
        state[head_y // self.grid_size, head_x // self.grid_size, 1] = 1

        # Mark food position with 1 in channel 2
        food_x, food_y = self.food
        state[food_y // self.grid_size, food_x // self.grid_size, 2] = 1

        return state

    def step(self, action):
        """
        Take a step in the game based on the action
        action: 0=up, 1=right, 2=down, 3=left
        Returns: (new_state, reward, done)
        """
        # Map action (0,1,2,3) to direction changes
        if action == 0 and self.direction != (0, self.grid_size):  # up
            self.direction = (0, -self.grid_size)
        elif action == 1 and self.direction != (-self.grid_size, 0):  # right
            self.direction = (self.grid_size, 0)
        elif action == 2 and self.direction != (0, -self.grid_size):  # down
            self.direction = (0, self.grid_size)
        elif action == 3 and self.direction != (self.grid_size, 0):  # left
            self.direction = (-self.grid_size, 0)

        # Move the snake: Add new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions with walls
        if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height):
            self.game_over = True
            return self._get_state(), -10, True

        # Check for collision with self
        if new_head in self.snake:
            self.game_over = True
            return self._get_state(), -10, True

        # Add new head to snake
        self.snake.insert(0, new_head)

        # Initialize reward
        reward = 0

        # Check if food was eaten
        if new_head == self.food:
            self.food = self._place_food()
            self.score += 1
            reward = 10
            self.steps_without_food = 0
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            reward = -0.1  # Small penalty for each move to encourage efficiency

            # Increment steps without food counter
            self.steps_without_food += 1

            # Check if snake is wandering too long without food
            if self.steps_without_food >= self.max_steps_without_food:
                self.game_over = True
                return self._get_state(), -5, True  # Penalty for timeout

        # Give more reward if snake gets closer to food
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Calculate Manhattan distance to food
        distance_to_food = abs(head_x - food_x) + abs(head_y - food_y)

        # Add small reward for getting closer to food
        if reward == -0.1:  # Only add distance reward if food wasn't eaten
            reward += 0.1 * (1.0 / (distance_to_food + 1))

        return self._get_state(), reward, self.game_over

    def play_human(self):
        """Run the game for human player"""
        self.init_pygame()
        self.reset()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.direction != (0, self.grid_size):
                        self.direction = (0, -self.grid_size)
                    elif event.key == pygame.K_RIGHT and self.direction != (-self.grid_size, 0):
                        self.direction = (self.grid_size, 0)
                    elif event.key == pygame.K_DOWN and self.direction != (0, -self.grid_size):
                        self.direction = (0, self.grid_size)
                    elif event.key == pygame.K_LEFT and self.direction != (self.grid_size, 0):
                        self.direction = (-self.grid_size, 0)

            # Move snake (similar to step function)
            head_x, head_y = self.snake[0]
            new_head = (head_x + self.direction[0], head_y + self.direction[1])

            # Check for collisions with walls
            if (new_head[0] < 0 or new_head[0] >= self.width or
                    new_head[1] < 0 or new_head[1] >= self.height or
                    new_head in self.snake):
                self.game_over = True
                self._display_game_over()
                pygame.time.wait(2000)  # Wait 2 seconds
                self.reset()
                continue

            # Add new head to snake
            self.snake.insert(0, new_head)

            # Check if food was eaten
            if new_head == self.food:
                self.food = self._place_food()
                self.score += 1
            else:
                # Remove tail if no food eaten
                self.snake.pop()

            # Draw everything
            self._draw()
            self.clock.tick(self.fps)

        pygame.quit()

    def _draw(self):
        """Draw the current game state"""
        self.screen.fill((0, 0, 0))

        # Draw snake
        for i, segment in enumerate(self.snake):
            color = (0, 255, 0) if i > 0 else (0, 200, 0)  # Darker green for head
            pygame.draw.rect(self.screen, color, pygame.Rect(segment[0], segment[1], self.grid_size, self.grid_size))

            # Add eyes to the head
            if i == 0:
                # Calculate eye positions based on direction
                eye_size = self.grid_size // 5
                offset = self.grid_size // 3

                if self.direction == (self.grid_size, 0):  # Right
                    eye1 = (segment[0] + self.grid_size - offset, segment[1] + offset)
                    eye2 = (segment[0] + self.grid_size - offset, segment[1] + self.grid_size - offset)
                elif self.direction == (-self.grid_size, 0):  # Left
                    eye1 = (segment[0] + offset, segment[1] + offset)
                    eye2 = (segment[0] + offset, segment[1] + self.grid_size - offset)
                elif self.direction == (0, -self.grid_size):  # Up
                    eye1 = (segment[0] + offset, segment[1] + offset)
                    eye2 = (segment[0] + self.grid_size - offset, segment[1] + offset)
                else:  # Down
                    eye1 = (segment[0] + offset, segment[1] + self.grid_size - offset)
                    eye2 = (segment[0] + self.grid_size - offset, segment[1] + self.grid_size - offset)

                pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(eye1[0], eye1[1], eye_size, eye_size))
                pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(eye2[0], eye2[1], eye_size, eye_size))

        # Draw food (red apple)
        pygame.draw.rect(self.screen, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1], self.grid_size, self.grid_size))

        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip()

    def _display_game_over(self):
        """Display game over message"""
        self.screen.fill((0, 0, 0))

        # Draw game over text
        game_over_text = self.font.render('Game Over!', True, (255, 0, 0))
        score_text = self.font.render(f'Final Score: {self.score}', True, (255, 255, 255))
        restart_text = self.font.render('Restarting...', True, (255, 255, 255))

        text_rect = game_over_text.get_rect(center=(self.width / 2, self.height / 2 - 50))
        score_rect = score_text.get_rect(center=(self.width / 2, self.height / 2))
        restart_rect = restart_text.get_rect(center=(self.width / 2, self.height / 2 + 50))

        self.screen.blit(game_over_text, text_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

    def render(self):
        """
        Render the current game state (called during AI play)
        """
        self.init_pygame()
        self._draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True


# If run directly, let a human play the game
if __name__ == "__main__":
    game = SnakeGame()
    game.play_human()