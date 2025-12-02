"""
Training script for One-Step Actor-Critic on Blackjack and CartPole environments.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

from environments import (
    create_blackjack_env,
    create_cartpole_env,
    preprocess_blackjack_state,
    preprocess_cartpole_state,
    get_state_dim,
    get_action_dim
)
from agent import OneStepActorCritic, OneStepActorCriticShared


def train_cartpole(
    num_episodes=1000,
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    hidden_dim=128,
    use_shared_network=False,
    render=False,
    verbose=True
):
    """
    Train One-Step Actor-Critic on CartPole environment.
    
    Args:
        num_episodes: Number of training episodes
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        gamma: Discount factor
        hidden_dim: Hidden layer dimension
        use_shared_network: Whether to use shared network architecture
        render: Whether to render environment
        verbose: Whether to print progress
    
    Returns:
        episode_rewards: List of episode rewards
        agent: Trained agent
    """
    env = create_cartpole_env()
    state_dim = get_state_dim('cartpole')
    action_dim = get_action_dim('cartpole')
    
    if use_shared_network:
        agent = OneStepActorCriticShared(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=actor_lr,
            gamma=gamma,
            hidden_dim=hidden_dim
        )
    else:
        agent = OneStepActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            hidden_dim=hidden_dim
        )
    
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_cartpole_state(state)
        episode_reward = 0
        agent.reset_episode()
        
        done = False
        truncated = False
        
        while not (done or truncated):
            if render:
                env.render()
            
            # Select action
            if use_shared_network:
                action, log_prob, value = agent.select_action(state)
            else:
                action, log_prob = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_cartpole_state(next_state)
            
            # Update agent
            if use_shared_network:
                agent.update(state, action, reward, next_state, done or truncated, log_prob, value)
            else:
                agent.update(state, action, reward, next_state, done or truncated, log_prob)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}")
            
            # Check if solved (CartPole is considered solved at 475 reward)
            if avg_reward >= 475:
                print(f"Environment solved in {episode + 1} episodes!")
                break
    
    env.close()
    return episode_rewards, agent


def train_blackjack(
    num_episodes=50000,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=1.0,  # No discounting for episodic tasks
    hidden_dim=64,
    use_shared_network=False,
    verbose=True
):
    """
    Train One-Step Actor-Critic on Blackjack environment.
    
    Args:
        num_episodes: Number of training episodes
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        gamma: Discount factor
        hidden_dim: Hidden layer dimension
        use_shared_network: Whether to use shared network architecture
        verbose: Whether to print progress
    
    Returns:
        episode_rewards: List of episode rewards
        win_rates: List of win rates over time
        agent: Trained agent
    """
    env = create_blackjack_env()
    state_dim = get_state_dim('blackjack')
    action_dim = get_action_dim('blackjack')
    
    if use_shared_network:
        agent = OneStepActorCriticShared(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=actor_lr,
            gamma=gamma,
            hidden_dim=hidden_dim
        )
    else:
        agent = OneStepActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            hidden_dim=hidden_dim
        )
    
    episode_rewards = []
    recent_rewards = deque(maxlen=1000)
    win_rates = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_blackjack_state(state)
        episode_reward = 0
        agent.reset_episode()
        
        done = False
        
        while not done:
            # Select action
            if use_shared_network:
                action, log_prob, value = agent.select_action(state)
            else:
                action, log_prob = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_blackjack_state(next_state)
            
            # Update agent
            if use_shared_network:
                agent.update(state, action, reward, next_state, done, log_prob, value)
            else:
                agent.update(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 5000 == 0:
            wins = sum(1 for r in recent_rewards if r > 0)
            draws = sum(1 for r in recent_rewards if r == 0)
            losses = sum(1 for r in recent_rewards if r < 0)
            win_rate = wins / len(recent_rewards)
            win_rates.append(win_rate)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Last 1000 games - Wins: {wins}, Draws: {draws}, Losses: {losses}")
            print(f"  Win rate: {win_rate:.2%}")
    
    env.close()
    return episode_rewards, win_rates, agent


def plot_results(rewards, title, save_path=None, window=100):
    """
    Plot training results.
    
    Args:
        rewards: List of episode rewards
        title: Plot title
        save_path: Path to save figure (optional)
        window: Moving average window
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', label=f'{window}-episode average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{title} - Episode Rewards')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'{title} - Cumulative Rewards')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train One-Step Actor-Critic')
    parser.add_argument('--env', type=str, default='cartpole',
                        choices=['cartpole', 'blackjack'],
                        help='Environment to train on')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes')
    parser.add_argument('--actor-lr', type=float, default=1e-3,
                        help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--shared-network', action='store_true',
                        help='Use shared network architecture')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save model')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save plot')
    
    args = parser.parse_args()
    
    if args.env == 'cartpole':
        print("Training on CartPole environment...")
        rewards, agent = train_cartpole(
            num_episodes=args.episodes,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            hidden_dim=args.hidden_dim,
            use_shared_network=args.shared_network,
            render=args.render
        )
        plot_results(rewards, 'CartPole', args.save_plot)
        
    elif args.env == 'blackjack':
        print("Training on Blackjack environment...")
        if args.episodes == 1000:  # Use default for Blackjack
            args.episodes = 50000
        rewards, win_rates, agent = train_blackjack(
            num_episodes=args.episodes,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=1.0,  # No discounting for Blackjack
            hidden_dim=args.hidden_dim,
            use_shared_network=args.shared_network
        )
        plot_results(rewards, 'Blackjack', args.save_plot, window=1000)
    
    if args.save_model:
        agent.save(args.save_model)
        print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
