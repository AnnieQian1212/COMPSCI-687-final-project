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
        'alpha_theta': 0.1,
        'alpha_w': 0.005,
        'gamma': 1,
        'feature_type': 'tabular', 
        'num_states': 500          
    },
}


if __name__ == '__main__':
    # Select environments to run
    # envs_to_run = ['cartpole', 'blackjack', 'frozenlake', 'taxi']
    envs_to_run = ['taxi']
    # envs_to_run = ['cartpole']
    # envs_to_run = ['blackjack']
    # envs_to_run = ['frozenlake']
    
    results = {}
    for env_name in envs_to_run:
        config = ENV_CONFIGS[env_name]
        results[env_name] = run_env(env_name, **config)
    
    plot_results(results)