import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from game import SnakeGame
from agent import DQNAgent


def train():
    # Configure training parameters
    grid_size = 20
    width = 400
    height = 400

    # Create game environment and agent
    game = SnakeGame(width=width, height=height, grid_size=grid_size)
    state_shape = game.reset().shape
    action_size = 4  # up, right, down, left

    # Initialize DQN agent
    agent = DQNAgent(state_shape, action_size)

    # Training parameters
    episodes = 1000  # Total episodes to train
    save_every = 100  # Save model every N episodes
    print_every = 1  # Print stats every episode

    # Create directory for saving models
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Training metrics
    scores = []
    avg_scores = []
    epsilons = []
    losses = []

    print("Starting training of AlphaSerpent...")
    print(f"State shape: {state_shape}, Action size: {action_size}")

    # Training loop
    try:
        for episode in range(episodes):
            # Reset environment
            state = game.reset()
            episode_reward = 0
            done = False
            steps = 0
            episode_loss = []

            # Episode loop
            while not done:
                # Get action from agent
                action = agent.act(state)

                # Take action in environment
                next_state, reward, done = game.step(action)

                # Store experience in agent's memory
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    episode_loss.append(loss)

                # Update state and metrics
                state = next_state
                episode_reward += reward
                steps += 1

            # Update target model periodically
            if episode % agent.update_target_freq == 0:
                agent.update_target_model()

            # Save model periodically
            if episode % save_every == 0 and episode > 0:
                agent.save(f"{models_dir}/alphaserpent_ep{episode}.h5")

            # Log metrics
            scores.append(game.score)
            epsilons.append(agent.epsilon)

            # Calculate and store average score over last 100 episodes
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            avg_scores.append(avg_score)

            # Calculate average loss for this episode
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)

            # Print progress
            if episode % print_every == 0:
                print(f"Episode: {episode}/{episodes}, Score: {game.score}, "
                      f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}, "
                      f"Steps: {steps}, Loss: {avg_loss:.4f}")

            # Save intermediate plots every 100 episodes
            if episode % 100 == 0 and episode > 0:
                plot_training_results(scores, avg_scores, epsilons, losses)

        # Save final model
        agent.save(f"{models_dir}/alphaserpent_final.h5")
        print("Training completed!")

        # Plot training progress
        plot_training_results(scores, avg_scores, epsilons, losses)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        agent.save(f"{models_dir}/alphaserpent_interrupted.h5")
        plot_training_results(scores, avg_scores, epsilons, losses)
        print("Model saved. Exiting.")


def plot_training_results(scores, avg_scores, epsilons, losses):
    """Plot and save training metrics"""
    # Create plots directory
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Set up figure with 3 subplots
    plt.figure(figsize=(15, 10))

    # Plot scores
    plt.subplot(3, 1, 1)
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Avg Score (100 ep)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend()

    # Plot epsilon decay
    plt.subplot(3, 1, 2)
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')

    # Plot loss
    plt.subplot(3, 1, 3)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Save and show plot
    plt.tight_layout()
    plt.savefig('plots/training_results.png')
    # Don't show plot during training to avoid blocking
    # plt.show()


if __name__ == "__main__":
    train()