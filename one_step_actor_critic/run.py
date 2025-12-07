"""
Main script to run experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from algo import OneStepActorCritic, train
from environments import make_env


def run_env(env_name, num_episodes, alpha_theta, alpha_w, gamma, feature_type, **feature_kwargs):
    print("=" * 50)
    print(f"{env_name.upper()} Environment")
    print("=" * 50)
    
    env = make_env(env_name)
    
    agent = OneStepActorCritic(
        num_actions=env.num_actions,
        state_dim=env.state_dim,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        gamma=gamma,
        feature_type=feature_type,
        state_bounds=env.state_bounds,
        **feature_kwargs
    )
    
    history = train(env, agent, num_episodes=num_episodes)
    env.close()
    return history


def plot_results(results_dict, window=100):
    num_envs = len(results_dict)
    fig, axes = plt.subplots(2, num_envs, figsize=(5 * num_envs, 10))
    
    if num_envs == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (name, history) in enumerate(results_dict.items()):
        rewards = history['rewards']
        # value_losses = history['value_losses']
        td_errors = history['td_errors']
        
        w = min(window, max(1, len(rewards) // 5))
        
        # Reward
        ax1 = axes[0, i]
        ma = np.convolve(rewards, np.ones(w)/w, mode='valid')
        ax1.plot(ma, color='blue')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title(f'{name} - Reward (Final: {np.mean(rewards[-100:]):.2f})')
        ax1.grid(True, alpha=0.3)
        
        # Value Loss (log scale)
        # ax2 = axes[1, i]
        # ma = np.convolve(value_losses, np.ones(w)/w, mode='valid')
        # ax2.plot(ma, color='red')
        # ax2.set_xlabel('Episode')
        # ax2.set_ylabel('Value Loss')
        # ax2.set_title(f'{name} - Value Loss (log scale)')
        # ax2.set_yscale('log')  # Use log scale
        # ax2.grid(True, alpha=0.3)
        
        # TD Error
        ax3 = axes[1, i]
        ma = np.convolve(td_errors, np.ones(w)/w, mode='valid')
        ax3.plot(ma, color='green')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('TD Error')
        ax3.set_title(f'{name} - TD Error')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("Figure saved to results.png")


# Environment configurations with appropriate features
ENV_CONFIGS = {
    'blackjack': {
        'num_episodes': 10000,
        'alpha_theta': 0.02,
        'alpha_w': 0.02,
        'gamma': 1.0,
        'feature_type': 'poly',
        'order': 3
    },
    'cartpole': {
        'num_episodes': 3000,
        'alpha_theta': 0.001,
        'alpha_w': 0.01,
        'gamma': 0.99,
        'feature_type': 'poly',
        'order': 2
    },
    'mountaincar': {
        'num_episodes': 5000,
        'alpha_theta': 0.01,
        'alpha_w': 0.1,
        'gamma': 0.99,
        'feature_type': 'tile',
        'num_tilings': 10,
        'num_tiles': 8
    },
    'acrobot': {
        'num_episodes': 10000,
        'alpha_theta': 0.05,
        'alpha_w': 0.2,
        'gamma': 0.99,
        'feature_type': 'tile',
        'num_tilings': 12,
        'num_tiles': 6
    },
    'lunarlander': {
        'num_episodes': 5000,
        'alpha_theta': 0.005,
        'alpha_w': 0.05,
        'gamma': 0.99,
        'feature_type': 'rbf',
        'centers_per_dim': 5,
        'sigma': 0.3
    },
    'cliffwalking': {
        'num_episodes': 3000,
        'alpha_theta': 0.01,
        'alpha_w': 0.05,
        'gamma': 0.99,
        'feature_type': 'tile',
        'num_tilings': 4,
        'num_tiles': 4
    },
    'frozenlake': {
        'num_episodes': 10000,
        'alpha_theta': 0.01,
        'alpha_w': 0.02, 
        'gamma': 0.99,
        'feature_type': 'tile',  
        'num_tilings': 4,
        'num_tiles': 4
    },
    'taxi': {
        'num_episodes': 1000,
        'alpha_theta': 0.01,
        'alpha_w': 0.05,  
        'gamma': 0.99,
        'feature_type': 'tile',  
        'num_tilings': 8,
        'num_tiles': 4
    },
}


if __name__ == '__main__':
    # Select environments to run
    # envs_to_run = ['cartpole', 'mountaincar', 'acrobot', 'blackjack', 'frozenlake', 'taxi']
    envs_to_run = ['cartpole', 'blackjack', 'frozenlake', 'taxi']
    
    results = {}
    for env_name in envs_to_run:
        config = ENV_CONFIGS[env_name]
        results[env_name] = run_env(env_name, **config)
    
    plot_results(results)