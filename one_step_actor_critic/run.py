"""
Main script to run experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from algo import OneStepActorCritic, train
from environments import BlackjackEnv, CartPoleEnv


def run_blackjack():
    print("=" * 50)
    print("Blackjack Environment")
    print("=" * 50)
    
    env = BlackjackEnv()
    # Blackjack state: (player_sum, dealer_card, usable_ace) -> 3 dims
    state_bounds = [(4, 21), (1, 10), (0, 1)]
    
    agent = OneStepActorCritic(
        num_actions=env.num_actions,
        state_dim=3,
        alpha_theta=0.0001,
        alpha_w=0.001,
        gamma=0.99,
        order=2,
        state_bounds=state_bounds
    )
    
    rewards = train(env, agent, num_episodes=50000)
    env.close()
    return rewards


def run_cartpole():
    print("\n" + "=" * 50)
    print("CartPole Environment")
    print("=" * 50)
    
    env = CartPoleEnv()
    # CartPole state: (position, velocity, angle, angular_velocity) -> 4 dims
    state_bounds = [(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2)]
    
    agent = OneStepActorCritic(
        num_actions=env.num_actions,
        state_dim=4,
        alpha_theta=0.001,
        alpha_w=0.01,
        gamma=0.99,
        order=2,
        state_bounds=state_bounds
    )
    
    rewards = train(env, agent, num_episodes=5000)
    env.close()
    return rewards


def plot_results(rewards_dict, window=100):
    plt.figure(figsize=(12, 4))
    
    for i, (name, rewards) in enumerate(rewards_dict.items()):
        plt.subplot(1, len(rewards_dict), i + 1)
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(ma)
        else:
            plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(name)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()
    print("Figure saved to results.png")


if __name__ == '__main__':
    results = {}

    # results['CartPole'] = run_cartpole()
    results['Blackjack'] = run_blackjack()
    
    plot_results(results)