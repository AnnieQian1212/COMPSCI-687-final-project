import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import os


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def compute_rolling_average(
    data: List[float],
    window_size: int = 10
) -> np.ndarray:
    """
    Compute rolling average of data.

    Args:
        data: List of values
        window_size: Size of the rolling window

    Returns:
        Array of rolling averages
    """
    if len(data) < window_size:
        return np.array(data)

    cumsum = np.cumsum(np.insert(data, 0, 0))
    rolling_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Pad the beginning with partial averages
    partial = [np.mean(data[:i+1]) for i in range(window_size - 1)]

    return np.concatenate([partial, rolling_avg])


def plot_learning_curve(
    rewards: List[float],
    title: str,
    xlabel: str = 'Episode',
    ylabel: str = 'Reward',
    window_size: int = 10,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a learning curve with smoothing.

    Args:
        rewards: List of episode rewards
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        window_size: Rolling average window size
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    episodes = np.arange(1, len(rewards) + 1)
    smoothed = compute_rolling_average(rewards, window_size)

    plt.plot(episodes, rewards, alpha=0.3, label='Raw rewards')
    plt.plot(episodes, smoothed, linewidth=2, label=f'Rolling avg (window={window_size})')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_hyperparameter_comparison(
    results: Dict[str, List[float]],
    param_name: str,
    title: str,
    ylabel: str = 'Average Reward',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot bar chart comparing different hyperparameter values.

    Args:
        results: Dictionary mapping parameter values to reward lists
        param_name: Name of the hyperparameter
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    labels = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]

    x = np.arange(len(labels))
    bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)

    plt.xlabel(param_name)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.1,
            f'{mean:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_learning_curves(
    results: Dict[str, List[float]],
    title: str,
    xlabel: str = 'Episode',
    ylabel: str = 'Reward',
    window_size: int = 10,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot multiple learning curves on the same graph.

    Args:
        results: Dictionary mapping labels to reward lists
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        window_size: Rolling average window size
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    for label, rewards in results.items():
        episodes = np.arange(1, len(rewards) + 1)
        smoothed = compute_rolling_average(rewards, window_size)
        plt.plot(episodes, smoothed, linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def print_statistics(
    name: str,
    rewards: List[float]
) -> Dict[str, float]:
    """
    Print and return statistics for a set of rewards.

    Args:
        name: Name of the experiment
        rewards: List of rewards

    Returns:
        Dictionary of statistics
    """
    stats = {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards)
    }

    print(f"\n{name} Statistics:")
    print(f"  Mean:   {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  Median: {stats['median']:.2f}")

    return stats


def create_summary_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Results Summary"
) -> str:
    """
    Create a formatted summary table.

    Args:
        results: Dictionary mapping config names to stats dictionaries
        title: Table title

    Returns:
        Formatted table string
    """
    header = f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"
    header += f"{'Configuration':<25} {'Mean':>10} {'Std':>10} {'Max':>10}\n"
    header += "-" * 60

    rows = []
    for name, stats in results.items():
        row = f"{name:<25} {stats['mean']:>10.2f} {stats['std']:>10.2f} {stats['max']:>10.2f}"
        rows.append(row)

    return header + "\n" + "\n".join(rows) + "\n" + "=" * 60


class RandomAgent:
    """Random baseline agent."""

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        """
        Initialize random agent.

        Args:
            n_actions: Number of actions
            seed: Random seed
        """
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Any) -> int:
        """Select a random action."""
        return self.rng.integers(self.n_actions)
