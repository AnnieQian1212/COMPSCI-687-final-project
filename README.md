# One-Step Actor-Critic Implementation

Implementation of the REINFORCE with Baseline algorithm, the One-Step Actor-Critic algorithm for episodic tasks and the Monte Carlo Tree Search (MCTS) algorithm in multiple OpenAI Gym environments.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/COMPSCI-687-final-project.git
cd COMPSCI-687-final-project/one_step_actor_critic
```

### 2. Create virtual environment
```bash
conda create -n actor-critic python=3.10
conda activate actor-critic
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Supported Environments

- **CartPole**: Balance a pole on a cart
- **Blackjack**: Card game with discrete actions
- **FrozenLake**: Navigate slippery grid to goal
- **Taxi**: Pick up and drop off passengers
- **MountainCar**: Drive up a steep hill
- **Acrobot**: Swing up a two-link robot arm
- **LunarLander**: Land spacecraft safely
- **CliffWalking**: Navigate grid avoiding cliff

## Quick Start

### Run with default configurations
```bash
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
