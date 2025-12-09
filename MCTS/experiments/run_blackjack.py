import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

from mcts.mcts_agent import BlackjackMCTSAgent
from mcts.utils import (
    set_seeds,
    plot_learning_curve,
    plot_hyperparameter_comparison,
    plot_multiple_learning_curves,
    print_statistics,
    create_summary_table,
    RandomAgent
)


def run_blackjack_episode(
    env: gym.Env,
    agent: BlackjackMCTSAgent
) -> float:
    """
    Run a single Blackjack episode.

    Args:
        env: Blackjack environment
        agent: MCTS agent

    Returns:
        Episode reward (-1, 0, or 1)
    """
    obs, _ = env.reset()
    done = False
    reward = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return reward


def run_random_blackjack_episode(env: gym.Env, agent: RandomAgent) -> float:
    """Run a single Blackjack episode with random agent."""
    obs, _ = env.reset()
    done = False
    reward = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return reward


def compute_win_rate(rewards: List[float]) -> float:
    """
    Compute win rate from rewards.

    Args:
        rewards: List of episode rewards

    Returns:
        Win rate as percentage
    """
    wins = sum(1 for r in rewards if r > 0)
    return (wins / len(rewards)) * 100


def experiment_basic_performance(
    n_episodes: int = 1000,
    n_simulations: int = 100,
    exploration_c: float = 1.41,
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 1: Basic performance comparison for Blackjack.

    Args:
        n_episodes: Number of episodes to run
        n_simulations: MCTS simulations per decision
        exploration_c: UCB exploration constant
        seed: Random seed
        show_plots: Whether to show plots

    Returns:
        Dictionary with MCTS and random agent rewards
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Basic Performance on Blackjack-v1")
    print("=" * 60)

    set_seeds(seed)

    # Create environment and agents
    env = gym.make('Blackjack-v1')
    mcts_agent = BlackjackMCTSAgent(
        env=env,
        n_simulations=n_simulations,
        exploration_c=exploration_c,
        seed=seed
    )
    random_agent = RandomAgent(n_actions=2, seed=seed)

    # Run MCTS agent
    print(f"\nRunning MCTS agent ({n_simulations} simulations per action)...")
    mcts_rewards = []
    for ep in tqdm(range(n_episodes), desc="MCTS"):
        reward = run_blackjack_episode(env, mcts_agent)
        mcts_rewards.append(reward)

    # Run random agent
    print("\nRunning random agent...")
    random_rewards = []
    for ep in tqdm(range(n_episodes), desc="Random"):
        reward = run_random_blackjack_episode(env, random_agent)
        random_rewards.append(reward)

    env.close()

    # Print statistics
    mcts_stats = print_statistics("MCTS Agent", mcts_rewards)
    random_stats = print_statistics("Random Agent", random_rewards)

    # Print win rates
    mcts_win_rate = compute_win_rate(mcts_rewards)
    random_win_rate = compute_win_rate(random_rewards)
    print(f"\nMCTS Win Rate: {mcts_win_rate:.1f}%")
    print(f"Random Win Rate: {random_win_rate:.1f}%")

    # Plot learning curve (cumulative win rate)
    mcts_cumulative_wins = np.cumsum([1 if r > 0 else 0 for r in mcts_rewards])
    mcts_win_rates = mcts_cumulative_wins / np.arange(1, n_episodes + 1) * 100

    random_cumulative_wins = np.cumsum([1 if r > 0 else 0 for r in random_rewards])
    random_win_rates = random_cumulative_wins / np.arange(1, n_episodes + 1) * 100

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.plot(mcts_win_rates, label='MCTS', linewidth=2)
    plt.plot(random_win_rates, label='Random', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Win Rate (%)')
    plt.title('Blackjack-v1: Cumulative Win Rate Over Episodes')
    plt.legend()
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/blackjack_win_rate.png', dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    return {'mcts': mcts_rewards, 'random': random_rewards}


def experiment_hyperparameter_simulations(
    n_episodes: int = 500,
    simulation_values: List[int] = [25, 50, 100, 200, 500],
    exploration_c: float = 1.41,
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 2a: Vary number of simulations for Blackjack.

    Args:
        n_episodes: Number of episodes per configuration
        simulation_values: List of simulation counts to test
        exploration_c: UCB exploration constant
        seed: Random seed
        show_plots: Whether to show plots

    Returns:
        Dictionary mapping simulation count to rewards
    """
    print("\n" + "=" * 60)
    print("Experiment 2a: Hyperparameter Study - n_simulations (Blackjack)")
    print("=" * 60)

    results = {}
    win_rates = {}

    for n_sim in simulation_values:
        print(f"\nTesting n_simulations = {n_sim}...")
        set_seeds(seed)

        env = gym.make('Blackjack-v1')
        agent = BlackjackMCTSAgent(
            env=env,
            n_simulations=n_sim,
            exploration_c=exploration_c,
            seed=seed
        )

        rewards = []
        for _ in tqdm(range(n_episodes), desc=f"n_sim={n_sim}"):
            reward = run_blackjack_episode(env, agent)
            rewards.append(reward)

        results[str(n_sim)] = rewards
        win_rates[str(n_sim)] = compute_win_rate(rewards)
        print_statistics(f"n_simulations={n_sim}", rewards)
        print(f"Win Rate: {win_rates[str(n_sim)]:.1f}%")

        env.close()

    # Plot win rate comparison
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    labels = list(win_rates.keys())
    values = list(win_rates.values())

    bars = plt.bar(labels, values, alpha=0.8)
    plt.axhline(y=28, color='r', linestyle='--', label='Random baseline (~28%)')

    plt.xlabel('Number of Simulations')
    plt.ylabel('Win Rate (%)')
    plt.title('Blackjack: Win Rate vs. Number of MCTS Simulations')
    plt.legend()

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{val:.1f}%',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig('results/blackjack_nsim_comparison.png', dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    return results


def experiment_hyperparameter_exploration(
    n_episodes: int = 500,
    n_simulations: int = 100,
    exploration_values: List[float] = [0.5, 1.0, 1.41, 2.0],
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 2b: Vary exploration constant for Blackjack.

    Args:
        n_episodes: Number of episodes per configuration
        n_simulations: MCTS simulations per decision
        exploration_values: List of exploration constants to test
        seed: Random seed
        show_plots: Whether to show plots

    Returns:
        Dictionary mapping exploration constant to rewards
    """
    print("\n" + "=" * 60)
    print("Experiment 2b: Hyperparameter Study - exploration_c (Blackjack)")
    print("=" * 60)

    results = {}
    win_rates = {}

    for exp_c in exploration_values:
        print(f"\nTesting exploration_c = {exp_c}...")
        set_seeds(seed)

        env = gym.make('Blackjack-v1')
        agent = BlackjackMCTSAgent(
            env=env,
            n_simulations=n_simulations,
            exploration_c=exp_c,
            seed=seed
        )

        rewards = []
        for _ in tqdm(range(n_episodes), desc=f"c={exp_c}"):
            reward = run_blackjack_episode(env, agent)
            rewards.append(reward)

        results[str(exp_c)] = rewards
        win_rates[str(exp_c)] = compute_win_rate(rewards)
        print_statistics(f"exploration_c={exp_c}", rewards)
        print(f"Win Rate: {win_rates[str(exp_c)]:.1f}%")

        env.close()

    # Plot comparison
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    labels = list(win_rates.keys())
    values = list(win_rates.values())

    bars = plt.bar(labels, values, alpha=0.8)
    plt.axhline(y=28, color='r', linestyle='--', label='Random baseline (~28%)')

    plt.xlabel('Exploration Constant (c)')
    plt.ylabel('Win Rate (%)')
    plt.title('Blackjack: Win Rate vs. UCB Exploration Constant')
    plt.legend()

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{val:.1f}%',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig('results/blackjack_exploration_comparison.png', dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    return results


def run_all_blackjack_experiments(
    show_plots: bool = True,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all Blackjack experiments.

    Args:
        show_plots: Whether to show plots
        seed: Random seed

    Returns:
        Dictionary with all experiment results
    """
    print("\n" + "#" * 60)
    print("# BLACKJACK-V1 EXPERIMENTS")
    print("#" * 60)

    results = {}

    # Experiment 1: Basic performance
    results['basic'] = experiment_basic_performance(
        n_episodes=1000,
        n_simulations=100,
        seed=seed,
        show_plots=show_plots
    )

    # Experiment 2a: Vary simulations
    results['simulations'] = experiment_hyperparameter_simulations(
        n_episodes=500,
        simulation_values=[25, 50, 100, 200, 500],
        seed=seed,
        show_plots=show_plots
    )

    # Experiment 2b: Vary exploration
    results['exploration'] = experiment_hyperparameter_exploration(
        n_episodes=500,
        exploration_values=[0.5, 1.0, 1.41, 2.0],
        seed=seed,
        show_plots=show_plots
    )

    # Create summary table
    summary = {}
    mcts_wr = compute_win_rate(results['basic']['mcts'])
    random_wr = compute_win_rate(results['basic']['random'])

    summary['MCTS (basic)'] = {
        'mean': np.mean(results['basic']['mcts']),
        'std': np.std(results['basic']['mcts']),
        'max': 1.0,
        'win_rate': mcts_wr
    }
    summary['Random'] = {
        'mean': np.mean(results['basic']['random']),
        'std': np.std(results['basic']['random']),
        'max': 1.0,
        'win_rate': random_wr
    }

    print("\n" + "=" * 60)
    print("Blackjack-v1 Results Summary")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Win Rate':>12} {'Mean Reward':>12}")
    print("-" * 60)
    print(f"{'MCTS (basic)':<20} {mcts_wr:>11.1f}% {summary['MCTS (basic)']['mean']:>12.3f}")
    print(f"{'Random':<20} {random_wr:>11.1f}% {summary['Random']['mean']:>12.3f}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    run_all_blackjack_experiments(show_plots=False, seed=42)
