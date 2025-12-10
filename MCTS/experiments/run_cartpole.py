import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

from mcts.mcts_agent import CartPoleMCTSAgent
from mcts.utils import (
    set_seeds,
    plot_learning_curve,
    plot_hyperparameter_comparison,
    plot_multiple_learning_curves,
    print_statistics,
    create_summary_table,
    RandomAgent
)


def run_cartpole_episode(
    env: gym.Env,
    agent: CartPoleMCTSAgent,
    render: bool = False
) -> float:
    """
    Run a single CartPole episode.

    Args:
        env: CartPole environment
        agent: MCTS agent
        render: Whether to render the environment

    Returns:
        Total episode reward
    """
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        if render:
            env.render()

        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def run_random_episode(env: gym.Env, agent: RandomAgent) -> float:
    """Run a single episode with random agent."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def experiment_basic_performance(
    n_episodes: int = 100,
    n_simulations: int = 50,
    exploration_c: float = 1.41,
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 1: Basic performance comparison.

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
    print("Experiment 1: Basic Performance on CartPole-v1")
    print("=" * 60)

    set_seeds(seed)

    # Create environment and agents
    env = gym.make('CartPole-v1')
    mcts_agent = CartPoleMCTSAgent(
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
        reward = run_cartpole_episode(env, mcts_agent)
        mcts_rewards.append(reward)

    # Run random agent
    print("\nRunning random agent...")
    random_rewards = []
    for ep in tqdm(range(n_episodes), desc="Random"):
        reward = run_random_episode(env, random_agent)
        random_rewards.append(reward)

    env.close()

    # Print statistics
    print_statistics("MCTS Agent", mcts_rewards)
    print_statistics("Random Agent", random_rewards)

    # Plot learning curve
    plot_learning_curve(
        mcts_rewards,
        title=f'MCTS Learning Curve on CartPole-v1 (n_sim={n_simulations})',
        save_path='results/cartpole_learning_curve.png',
        show=show_plots
    )

    return {'mcts': mcts_rewards, 'random': random_rewards}


def experiment_hyperparameter_simulations(
    n_episodes: int = 50,
    simulation_values: List[int] = [10, 25, 50, 100, 200],
    exploration_c: float = 1.41,
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 2a: Vary number of simulations.

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
    print("Experiment 2a: Hyperparameter Study - n_simulations")
    print("=" * 60)

    results = {}

    for n_sim in simulation_values:
        print(f"\nTesting n_simulations = {n_sim}...")
        set_seeds(seed)

        env = gym.make('CartPole-v1')
        agent = CartPoleMCTSAgent(
            env=env,
            n_simulations=n_sim,
            exploration_c=exploration_c,
            seed=seed
        )

        rewards = []
        for _ in tqdm(range(n_episodes), desc=f"n_sim={n_sim}"):
            reward = run_cartpole_episode(env, agent)
            rewards.append(reward)

        results[str(n_sim)] = rewards
        print_statistics(f"n_simulations={n_sim}", rewards)

        env.close()

    # Plot comparison
    plot_hyperparameter_comparison(
        results,
        param_name='Number of Simulations',
        title='CartPole: Performance vs. Number of MCTS Simulations',
        save_path='results/cartpole_nsim_comparison.png',
        show=show_plots
    )

    # Plot learning curves together
    labeled_results = {f"n_sim={k}": v for k, v in results.items()}
    plot_multiple_learning_curves(
        labeled_results,
        title='CartPole: Learning Curves for Different Simulation Counts',
        save_path='results/cartpole_nsim_curves.png',
        show=show_plots
    )

    return results


def experiment_hyperparameter_exploration(
    n_episodes: int = 50,
    n_simulations: int = 50,
    exploration_values: List[float] = [0.5, 1.0, 1.41, 2.0],
    seed: int = 42,
    show_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 2b: Vary exploration constant.

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
    print("Experiment 2b: Hyperparameter Study - exploration_c")
    print("=" * 60)

    results = {}

    for exp_c in exploration_values:
        print(f"\nTesting exploration_c = {exp_c}...")
        set_seeds(seed)

        env = gym.make('CartPole-v1')
        agent = CartPoleMCTSAgent(
            env=env,
            n_simulations=n_simulations,
            exploration_c=exp_c,
            seed=seed
        )

        rewards = []
        for _ in tqdm(range(n_episodes), desc=f"c={exp_c}"):
            reward = run_cartpole_episode(env, agent)
            rewards.append(reward)

        results[str(exp_c)] = rewards
        print_statistics(f"exploration_c={exp_c}", rewards)

        env.close()

    # Plot comparison
    plot_hyperparameter_comparison(
        results,
        param_name='Exploration Constant (c)',
        title='CartPole: Performance vs. UCB Exploration Constant',
        save_path='results/cartpole_exploration_comparison.png',
        show=show_plots
    )

    return results


def run_all_cartpole_experiments(
    show_plots: bool = True,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all CartPole experiments.

    Args:
        show_plots: Whether to show plots
        seed: Random seed

    Returns:
        Dictionary with all experiment results
    """
    print("\n" + "#" * 60)
    print("# CARTPOLE-V1 EXPERIMENTS")
    print("#" * 60)

    results = {}

    # Experiment 1: Basic performance
    results['basic'] = experiment_basic_performance(
        n_episodes=100,
        n_simulations=50,
        seed=seed,
        show_plots=show_plots
    )

    # Experiment 2a: Vary simulations
    results['simulations'] = experiment_hyperparameter_simulations(
        n_episodes=50,
        simulation_values=[10, 25, 50, 100, 200],
        seed=seed,
        show_plots=show_plots
    )

    # Experiment 2b: Vary exploration
    results['exploration'] = experiment_hyperparameter_exploration(
        n_episodes=50,
        exploration_values=[0.5, 1.0, 1.41, 2.0],
        seed=seed,
        show_plots=show_plots
    )

    # Create summary table
    summary = {}
    summary['MCTS (basic)'] = print_statistics("MCTS Basic", results['basic']['mcts'])
    summary['Random'] = print_statistics("Random", results['basic']['random'])

    for n_sim, rewards in results['simulations'].items():
        summary[f'n_sim={n_sim}'] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'max': np.max(rewards)
        }

    print(create_summary_table(summary, "CartPole-v1 Results Summary"))

    return results


if __name__ == '__main__':
    run_all_cartpole_experiments(show_plots=False, seed=42)
