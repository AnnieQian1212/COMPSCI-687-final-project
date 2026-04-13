This project implents three reinforcement learning algorithms: REINFORCE with Baseline (Wanqi Li), One-Step Actor-Critic(Aichen Qian), and Monte Carlo Tree Search (Linfeng Lyu). Each algorithm is implemented in its own directory with detailed instructions below. Group has three members: Wanqi Li, Aichen Qian, Linfeng Lyu.

# Setup


## 1. Create virtual environment
```bash
conda create -n actor-critic python=3.10
conda activate actor-critic
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Supported Environments

- **CartPole**: Balance a pole on a cart
- **Blackjack**: Card game with discrete actions
- **FrozenLake**: Navigate slippery grid to goal
- **Taxi**: Pick up and drop off passengers

# REINFORCE with baseline
Implementation the REINFORCE with baseline algorithm for episodic tasks in multiple OpenAI Gym environments.


## Usage
```bash
cd REINFORCE_with_baseline
python parallel.py
```

## Hyperparameters
- `alpha`: Policy step size
- `alpha_w`: Value function step size  
- `gamma`: Discount factor
- `policy_neurons_per_layer`: policy network architecture
- `value_neurons_per_layer`: value function network architecture

[See content details here](https://github.com/AnnieQian1212/COMPSCI-687-final-project/tree/main/REINFORCE_with_baseline)

# One-Step Actor-Critic Implementation

Implementation the One-Step Actor-Critic algorithm for episodic tasks in multiple OpenAI Gym environments.



## Quick Start

### Run with default configurations
```bash
cd one_step_actor_critic
python run.py
```

### Run hyperparameter tuning
```bash
python tune_hyperparams.py
```

## Key Hyperparameters

- `alpha_theta`: Policy learning rate
- `alpha_w`: Value function learning rate  
- `gamma`: Discount factor
- `feature_type`: Feature representation type
- Feature-specific params (order, num_tilings, centers, etc.)




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
```



## Usage

Run all experiments:
```bash
cd MCTS
```
```bash
python main.py --all
```

Run specific experiments:
```bash
python main.py --cartpole      # CartPole only
python main.py --blackjack     # Blackjack only
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



## Hyperparameters

- `n_simulations`: Number of MCTS iterations per decision (default: 100)
- `exploration_c`: UCB exploration constant (default: 1.41)

## Results

Results are saved to the `results/` directory.

