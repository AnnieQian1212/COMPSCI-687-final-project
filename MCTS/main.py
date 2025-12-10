#!/usr/bin/env python3
"""
MCTS Agent for Gymnasium Environments
Main entry point for running experiments
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_cartpole import run_all_cartpole_experiments
from experiments.run_blackjack import run_all_blackjack_experiments
from experiments.compare import run_all_comparisons


def main():
    """Main entry point for MCTS experiments."""
    parser = argparse.ArgumentParser(
        description='MCTS Agent for Gymnasium Environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                Run all experiments
  python main.py --cartpole           Run CartPole experiments only
  python main.py --blackjack          Run Blackjack experiments only
  python main.py --compare            Run comparison experiments only
  python main.py --all --show-plots   Run all experiments with plot display
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )
    parser.add_argument(
        '--cartpole',
        action='store_true',
        help='Run CartPole experiments'
    )
    parser.add_argument(
        '--blackjack',
        action='store_true',
        help='Run Blackjack experiments'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparison experiments'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # If no specific experiment selected, run all
    run_all = args.all or not (args.cartpole or args.blackjack or args.compare)

    print("=" * 60)
    print("MCTS AGENT FOR GYMNASIUM ENVIRONMENTS")
    print("=" * 60)
    print(f"Random seed: {args.seed}")
    print(f"Show plots: {args.show_plots}")
    print("=" * 60)

    results = {}

    # Run selected experiments
    if run_all or args.cartpole:
        print("\n>>> Running CartPole experiments...")
        results['cartpole'] = run_all_cartpole_experiments(
            show_plots=args.show_plots,
            seed=args.seed
        )

    if run_all or args.blackjack:
        print("\n>>> Running Blackjack experiments...")
        results['blackjack'] = run_all_blackjack_experiments(
            show_plots=args.show_plots,
            seed=args.seed
        )

    if run_all or args.compare:
        print("\n>>> Running comparison experiments...")
        results['comparison'] = run_all_comparisons(
            show_plots=args.show_plots,
            seed=args.seed
        )

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print("Results saved to: results/")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
