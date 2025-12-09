import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict

from mcts.mcts_agent import CartPoleMCTSAgent, BlackjackMCTSAgent
from mcts.utils import set_seeds, RandomAgent


def run_cartpole_comparison(
    n_episodes: int = 100,
    n_simulations: int = 100,
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, Dict]:
    """
    Run comparison between MCTS and random agent on CartPole.

    Args:
        n_episodes: Number of episodes
        n_simulations: MCTS simulations per action
        seed: Random seed
        show_plots: Whether to show plots

    Returns:
        Comparison results
    """
    print("\n" + "=" * 60)
    print("CartPole-v1: MCTS vs Random Agent Comparison")
    print("=" * 60)

    set_seeds(seed)
    env = gym.make('CartPole-v1')

    # MCTS Agent
    mcts_agent = CartPoleMCTSAgent(
        env=env,
        n_simulations=n_simulations,
        exploration_c=1.41,
        seed=seed
    )

    # Random Agent
    random_agent = RandomAgent(n_actions=2, seed=seed)

    # Run MCTS
    mcts_rewards = []
    print("\nRunning MCTS agent...")
    for _ in tqdm(range(n_episodes), desc="MCTS"):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = mcts_agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        mcts_rewards.append(total_reward)

    # Run Random
    random_rewards = []
    print("\nRunning Random agent...")
    for _ in tqdm(range(n_episodes), desc="Random"):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = random_agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        random_rewards.append(total_reward)

    env.close()

    # Results
    results = {
        'mcts': {
            'rewards': mcts_rewards,
            'mean': np.mean(mcts_rewards),
            'std': np.std(mcts_rewards),
            'max': np.max(mcts_rewards)
        },
        'random': {
            'rewards': random_rewards,
            'mean': np.mean(random_rewards),
            'std': np.std(random_rewards),
            'max': np.max(random_rewards)
        }
    }

    print(f"\nCartPole Results (n_sim={n_simulations}):")
    print(f"  MCTS:   {results['mcts']['mean']:.1f} ± {results['mcts']['std']:.1f}")
    print(f"  Random: {results['random']['mean']:.1f} ± {results['random']['std']:.1f}")
    print(f"  Improvement: {(results['mcts']['mean'] / results['random']['mean'] - 1) * 100:.1f}%")

    return results


def run_blackjack_comparison(
    n_episodes: int = 1000,
    n_simulations: int = 100,
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, Dict]:
    """
    Run comparison between MCTS and random agent on Blackjack.

    Args:
        n_episodes: Number of episodes
        n_simulations: MCTS simulations per action
        seed: Random seed
        show_plots: Whether to show plots

    Returns:
        Comparison results
    """
    print("\n" + "=" * 60)
    print("Blackjack-v1: MCTS vs Random Agent Comparison")
    print("=" * 60)

    set_seeds(seed)
    env = gym.make('Blackjack-v1')

    # MCTS Agent
    mcts_agent = BlackjackMCTSAgent(
        env=env,
        n_simulations=n_simulations,
        exploration_c=1.41,
        seed=seed
    )

    # Random Agent
    random_agent = RandomAgent(n_actions=2, seed=seed)

    # Run MCTS
    mcts_rewards = []
    print("\nRunning MCTS agent...")
    for _ in tqdm(range(n_episodes), desc="MCTS"):
        obs, _ = env.reset()
        done = False
        reward = 0
        while not done:
            action = mcts_agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        mcts_rewards.append(reward)

    # Run Random
    random_rewards = []
    print("\nRunning Random agent...")
    for _ in tqdm(range(n_episodes), desc="Random"):
        obs, _ = env.reset()
        done = False
        reward = 0
        while not done:
            action = random_agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        random_rewards.append(reward)

    env.close()

    # Compute win rates
    mcts_wins = sum(1 for r in mcts_rewards if r > 0)
    random_wins = sum(1 for r in random_rewards if r > 0)

    results = {
        'mcts': {
            'rewards': mcts_rewards,
            'mean': np.mean(mcts_rewards),
            'win_rate': mcts_wins / n_episodes * 100
        },
        'random': {
            'rewards': random_rewards,
            'mean': np.mean(random_rewards),
            'win_rate': random_wins / n_episodes * 100
        }
    }

    print(f"\nBlackjack Results (n_sim={n_simulations}):")
    print(f"  MCTS Win Rate:   {results['mcts']['win_rate']:.1f}%")
    print(f"  Random Win Rate: {results['random']['win_rate']:.1f}%")
    print(f"  Improvement: +{results['mcts']['win_rate'] - results['random']['win_rate']:.1f}%")

    return results


def create_comparison_plots(
    cartpole_results: Dict,
    blackjack_results: Dict,
    show_plots: bool = True
) -> None:
    """
    Create comparison plots for both environments.

    Args:
        cartpole_results: CartPole comparison results
        blackjack_results: Blackjack comparison results
        show_plots: Whether to show plots
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CartPole comparison
    ax1 = axes[0]
    agents = ['MCTS', 'Random']
    cp_means = [cartpole_results['mcts']['mean'], cartpole_results['random']['mean']]
    cp_stds = [cartpole_results['mcts']['std'], cartpole_results['random']['std']]

    bars1 = ax1.bar(agents, cp_means, yerr=cp_stds, capsize=5, alpha=0.8,
                   color=['#2ecc71', '#e74c3c'])
    ax1.set_ylabel('Average Reward')
    ax1.set_title('CartPole-v1: MCTS vs Random')
    ax1.axhline(y=400, color='blue', linestyle='--', alpha=0.5, label='Target (400)')
    ax1.legend()

    for bar, mean in zip(bars1, cp_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')

    # Blackjack comparison
    ax2 = axes[1]
    bj_rates = [blackjack_results['mcts']['win_rate'], blackjack_results['random']['win_rate']]

    bars2 = ax2.bar(agents, bj_rates, alpha=0.8, color=['#2ecc71', '#e74c3c'])
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Blackjack-v1: MCTS vs Random')
    ax2.axhline(y=40, color='blue', linestyle='--', alpha=0.5, label='Target (40%)')
    ax2.legend()

    for bar, rate in zip(bars2, bj_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/mcts_vs_random_comparison.png', dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close()


def run_all_comparisons(
    show_plots: bool = True,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all comparison experiments.

    Args:
        show_plots: Whether to show plots
        seed: Random seed

    Returns:
        All comparison results
    """
    print("\n" + "#" * 60)
    print("# MCTS VS RANDOM AGENT COMPARISON")
    print("#" * 60)

    # Run comparisons
    cartpole_results = run_cartpole_comparison(
        n_episodes=100,
        n_simulations=100,
        seed=seed,
        show_plots=show_plots
    )

    blackjack_results = run_blackjack_comparison(
        n_episodes=1000,
        n_simulations=100,
        seed=seed,
        show_plots=show_plots
    )

    # Create comparison plots
    create_comparison_plots(cartpole_results, blackjack_results, show_plots)

    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Environment':<15} {'Metric':<15} {'MCTS':>12} {'Random':>12} {'Target':>12}")
    print("-" * 60)
    print(f"{'CartPole':<15} {'Avg Reward':<15} {cartpole_results['mcts']['mean']:>12.1f} {cartpole_results['random']['mean']:>12.1f} {'>400':>12}")
    print(f"{'Blackjack':<15} {'Win Rate (%)':<15} {blackjack_results['mcts']['win_rate']:>12.1f} {blackjack_results['random']['win_rate']:>12.1f} {'>40%':>12}")
    print("=" * 60)

    # Check success criteria
    print("\nSuccess Criteria Check:")
    cp_success = cartpole_results['mcts']['mean'] > 400
    bj_success = blackjack_results['mcts']['win_rate'] > 40

    print(f"  CartPole (avg > 400): {'PASS' if cp_success else 'FAIL'}")
    print(f"  Blackjack (win rate > 40%): {'PASS' if bj_success else 'FAIL'}")

    return {
        'cartpole': cartpole_results,
        'blackjack': blackjack_results
    }


if __name__ == '__main__':
    run_all_comparisons(show_plots=False, seed=42)
