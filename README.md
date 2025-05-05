# AlphaSerpent

AlphaSerpent is a Deep Q-Learning implementation of an AI that learns to play the classic Snake game.
## Project Overview

This project implements:
- A Snake game environment with Pygame
- A Deep Q-Network (DQN) agent that learns to play Snake
- Training pipeline for the agent
- Visualisation for the training progress
- The ability to watch the trained AI play

## Project Structure

- `game.py`: Snake game implementation with both visual and agent interfaces
- `agent.py`: DQN agent implementation using TensorFlow/Keras
- `train.py`: Training script for the AI
- `play_ai.py`: Script to watch the trained AI play the game
- `models/`: Directory to store trained model weights

## How to Run

### Setup
```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib pygame
```

### Training the AI
```bash
python train.py
```

### Watching the AI Play
```bash
python play_ai.py --model models/alphaserpent_final.h5 --delay 0.1
```

### Playing the Game Yourself
The game environment also supports human play:
```python
from game import SnakeGame
game = SnakeGame()
game.play_human()
```

---

*Note: This project is based on a reinforcement learning tutorial and serves as a demonstration of applying deep Q-learning to game playing.* 
