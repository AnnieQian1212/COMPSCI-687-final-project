import numpy as np
import matplotlib.pyplot as plt
from algo import OneStepActorCritic, train
from environments import make_env


def run_config(env_name, config, config_name):
    """Run single configuration"""
    env = make_env(env_name)
    agent = OneStepActorCritic(
        num_actions=env.num_actions,
        state_dim=env.state_dim,
        alpha_theta=config['alpha_theta'],
        alpha_w=config['alpha_w'],
        gamma=config['gamma'],
        feature_type=config['feature_type'],
        state_bounds=env.state_bounds,
        **{k: v for k, v in config.items() if k not in ['alpha_theta', 'alpha_w', 'gamma', 'feature_type', 'num_episodes']}
    )
    
    history = train(env, agent, num_episodes=config['num_episodes'], verbose=False)
    env.close()
    
    final_reward = np.mean(history['rewards'][-100:])
    print(f"{config_name}: {final_reward:.2f}")
    
    return history, final_reward


def plot_comparison(results, env_name):
    """Plot 4 configs comparison"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    
    for idx, (name, data) in enumerate(results.items()):
        rewards = data['history']['rewards']
        td_errors = data['history']['td_errors']
        
        window = 100
        ma_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ma_td = np.convolve(td_errors, np.ones(window)/window, mode='valid')
        
        # Reward plot
        ax1 = axes[0, idx]
        ax1.plot(ma_reward, linewidth=2, color='blue')
        ax1.set_title(f"{name}\nFinal: {data['final_reward']:.2f}", fontweight='bold', fontsize=10)
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # TD Error plot
        ax2 = axes[1, idx]
        ax2.plot(ma_td, linewidth=2, color='green')
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('TD Error')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{env_name.upper()} - 4 Configurations Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{env_name}_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {env_name}_tuning.png")


# ==================== 4 Configs per Environment ====================

CARTPOLE_CONFIGS = {
    'Config 1: Poly Order 2 (baseline)': {
        'num_episodes': 2000,
        'alpha_theta': 0.001,
        'alpha_w': 0.01,
        'gamma': 0.99,
        'feature_type': 'poly',
        'order': 2
    },
    'Config 2: Poly Order 3': {
        'num_episodes': 2000,
        'alpha_theta': 0.002,
        'alpha_w': 0.02,
        'gamma': 0.99,
        'feature_type': 'poly',
        'order': 3
    },
    'Config 3: Fourier Order 2': {
        'num_episodes': 2000,
        'alpha_theta': 0.001,
        'alpha_w': 0.01,
        'gamma': 0.99,
        'feature_type': 'fourier',
        'order': 2
    },
    'Config 4: RBF': {
        'num_episodes': 2000,
        'alpha_theta': 0.001,
        'alpha_w': 0.01,
        'gamma': 0.99,
        'feature_type': 'rbf',
        'centers_per_dim': 4,
        'sigma': 0.3
    }
}

BLACKJACK_CONFIGS = {
    'Config 1: Poly Order 2 (High LR)': {
        'num_episodes': 50000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 1.0,
        'feature_type': 'poly',
        'order': 2
    },
    'Config 2: Poly Order 3 (High LR)': {
        'num_episodes': 50000,
        'alpha_theta': 0.1,
        'alpha_w': 0.2,
        'gamma': 1.0,
        'feature_type': 'poly',
        'order': 3
    },
    'Config 3: Tile Coding (Fast)': {
        'num_episodes': 50000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 1.0,
        'feature_type': 'tile',
        'num_tilings': 4,
        'num_tiles': 3
    },
    'Config 4: Fourier (Aggressive)': {
        'num_episodes': 50000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 1.0,
        'feature_type': 'fourier',
        'order': 2
    }
}

FROZENLAKE_CONFIGS = {
    'Config 1: Tile 4x4': {
        'num_episodes': 5000,
        'alpha_theta': 0.01,
        'alpha_w': 0.02,
        'gamma': 0.99,
        'feature_type': 'tile',
        'num_tilings': 4,
        'num_tiles': 4
    },
    'Config 2: Tile 6x6': {
        'num_episodes': 5000,
        'alpha_theta': 0.02,
        'alpha_w': 0.05,
        'gamma': 0.99,
        'feature_type': 'tile',
        'num_tilings': 6,
        'num_tiles': 4
    },
    'Config 3: Coarse Coding': {
        'num_episodes': 5000,
        'alpha_theta': 0.01,
        'alpha_w': 0.02,
        'gamma': 0.99,
        'feature_type': 'coarse',
        'num_circles': 50,
        'radius': 0.3
    },
    'Config 4: Poly Order 2': {
        'num_episodes': 5000,
        'alpha_theta': 0.01,
        'alpha_w': 0.02,
        'gamma': 0.99,
        'feature_type': 'poly',
        'order': 2
    }
}

TAXI_CONFIGS = {
    'Config 1: Tile 6x3 (Fast)': {
        'num_episodes': 3000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 0.99,
        'feature_type': 'tile',
        'num_tilings': 6,
        'num_tiles': 3
    },
    'Config 2: Poly Order 2 (Fast)': {
        'num_episodes': 3000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 0.99,
        'feature_type': 'poly',
        'order': 2
    },
    'Config 3: Coarse (Fast)': {
        'num_episodes': 3000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 0.99,
        'feature_type': 'coarse',
        'num_circles': 50,
        'radius': 0.4
    },
    'Config 4: RBF (Fast)': {
        'num_episodes': 3000,
        'alpha_theta': 0.05,
        'alpha_w': 0.1,
        'gamma': 0.99,
        'feature_type': 'rbf',
        'centers_per_dim': 3,
        'sigma': 0.4
    }
}


# ==================== Main ====================

if __name__ == '__main__':
    
    # CartPole
    print("\n" + "="*60)
    print("CARTPOLE")
    print("="*60)
    results = {}
    for name, config in CARTPOLE_CONFIGS.items():
        history, final_reward = run_config('cartpole', config, name)
        results[name] = {'history': history, 'final_reward': final_reward}
    plot_comparison(results, 'cartpole')
    
    # Blackjack
    print("\n" + "="*60)
    print("BLACKJACK")
    print("="*60)
    results = {}
    for name, config in BLACKJACK_CONFIGS.items():
        history, final_reward = run_config('blackjack', config, name)
        results[name] = {'history': history, 'final_reward': final_reward}
    plot_comparison(results, 'blackjack')
    
    # FrozenLake
    print("\n" + "="*60)
    print("FROZENLAKE")
    print("="*60)
    results = {}
    for name, config in FROZENLAKE_CONFIGS.items():
        history, final_reward = run_config('frozenlake', config, name)
        results[name] = {'history': history, 'final_reward': final_reward}
    plot_comparison(results, 'frozenlake')
    
    # Taxi
    print("\n" + "="*60)
    print("TAXI")
    print("="*60)
    results = {}
    for name, config in TAXI_CONFIGS.items():
        history, final_reward = run_config('taxi', config, name)
        results[name] = {'history': history, 'final_reward': final_reward}
    plot_comparison(results, 'taxi')
    
    print("\n" + "="*60)
    print("TUNING COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  - cartpole_tuning.png")
    print("  - blackjack_tuning.png")
    print("  - frozenlake_tuning.png")
    print("  - taxi_tuning.png")
    print("="*60)