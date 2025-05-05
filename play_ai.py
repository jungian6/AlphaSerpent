import argparse
import time
from game import SnakeGame
from agent import DQNAgent


def play_ai(model_path, delay=0.1):
    """
    Watch the trained AI play Snake

    Args:
        model_path: Path to the trained model weights
        delay: Delay between steps (seconds) to make it watchable
    """
    # Set up game with the same parameters used during training
    grid_size = 20
    width = 400
    height = 400
    game = SnakeGame(width=width, height=height, grid_size=grid_size)

    # Initialize agent
    state_shape = game.reset().shape
    action_size = 4
    agent = DQNAgent(state_shape, action_size)

    # Load trained weights
    print(f"Loading model from {model_path}")
    agent.load(model_path)

    # Disable exploration
    agent.epsilon = 0

    # Play until user quits
    max_games = 10
    games_played = 0
    total_score = 0

    while games_played < max_games:
        # Reset game
        state = game.reset()
        done = False
        step_count = 0

        # Game loop
        while not done:
            # Get AI action
            action = agent.act(state, training=False)

            # Take action
            next_state, _, done = game.step(action)
            state = next_state
            step_count += 1

            # Render game
            if not game.render():
                return  # Exit if window closed

            # Add delay to make it watchable
            time.sleep(delay)

        # Game over
        games_played += 1
        total_score += game.score
        print(f"Game {games_played}: Score = {game.score}, Steps = {step_count}")

        # Wait before starting next game
        time.sleep(1)

    print(f"Average score over {games_played} games: {total_score / games_played:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Watch AlphaSerpent AI play Snake')
    parser.add_argument('--model', type=str, default='models/alphaserpent_final.h5',
                        help='Path to model weights file')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps (seconds)')
    args = parser.parse_args()

    play_ai(args.model, args.delay)