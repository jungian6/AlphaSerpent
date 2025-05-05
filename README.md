# AlphaSerpent

AlphaSerpent is a Deep Q-Learning implementation of an AI that learns to play the classic Snake game. This project is based on a reinforcement learning tutorial and demonstrates the application of deep learning to game playing.

## What it does

- Snake game built with Pygame
- AI that learns to play using deep Q-learning
- Training visualisations to track progress

## Files

- `game.py` - Snake game implementation
- `agent.py` - DQN agent using TensorFlow
- `train.py` - Trains the AI
- `play_ai.py` - Watch the AI play

## Quick Start

```bash
# Install requirements
pip install -r requirements.txt

# Train the AI 
python train.py

# Watch it play
python play_ai.py
```

You can also play yourself:
```python
from game import SnakeGame
game = SnakeGame()
game.play_human()
```

## Google Colab Version

If you don't want to melt your own GPU, use Google's TPUs instead:
[Run on Google Colab](https://colab.research.google.com/drive/1ALm56h9ngUZCf4Q78PnE___TpokcuIlR?usp=sharing)

This was a fun project to learn about reinforcement learning. The AI starts out terrible but gets pretty good after training! 