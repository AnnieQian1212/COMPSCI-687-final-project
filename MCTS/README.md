# MCTS Agent for Gymnasium Environments

Monte Carlo Tree Search (MCTS) implementation for solving Gymnasium environments, demonstrating MCTS on both deterministic (CartPole) and stochastic (Blackjack) environments.

## Project Structure

```
mcts_project/
├── mcts/
│   ├── __init__.py
│   ├── node.py           # MCTSNode class
│   ├── mcts_agent.py     # Main MCTS algorithm
│   └── utils.py          # Helper functions
├── experiments/
│   ├── run_cartpole.py   # CartPole experiments
│   ├── run_blackjack.py  # Blackjack experiments
│   └── compare.py        # Comparison experiments
├── results/              # Generated plots and data
├── main.py               # Main entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:
```bash
python main.py --all
```

Run specific experiments:
```bash
python main.py --cartpole      # CartPole only
python main.py --blackjack     # Blackjack only
python main.py --compare       # Comparison only
```

Show plots interactively:
```bash
python main.py --all --show-plots
```

Set random seed:
```bash
python main.py --all --seed 123
```

## MCTS Algorithm

The implementation follows the four phases of MCTS:

1. **Selection**: Traverse tree using UCB1 formula
   ```
   UCB(n) = Q(n)/N(n) + c * sqrt(ln(N_parent) / N(n))
   ```

2. **Expansion**: Add new child node for unexplored action

3. **Simulation/Rollout**: Random policy until terminal state

4. **Backpropagation**: Update visit counts and rewards up the tree

## Environments

### CartPole-v1 (Deterministic)
- Discrete actions: 0 (left), 1 (right)
- Uses action-sequence replay approach
- Target: Average reward > 400

### Blackjack-v1 (Stochastic)
- Discrete actions: 0 (stick), 1 (hit)
- Uses flat MCTS approach
- Target: Win rate > 40%

## Experiments

1. **Basic Performance**: Compare MCTS vs random agent
2. **Hyperparameter Study**: Vary n_simulations and exploration_c
3. **Learning Curves**: Track performance over episodes

## Hyperparameters

- `n_simulations`: Number of MCTS iterations per decision (default: 100)
- `exploration_c`: UCB exploration constant (default: 1.41)

## Results

Results are saved to the `results/` directory:
- `cartpole_learning_curve.png`
- `cartpole_nsim_comparison.png`
- `cartpole_exploration_comparison.png`
- `blackjack_win_rate.png`
- `blackjack_nsim_comparison.png`
- `mcts_vs_random_comparison.png`
