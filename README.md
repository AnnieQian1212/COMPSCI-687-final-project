# COMPSCI-687 Final Project: One-Step Actor-Critic

Implementation of the **One-Step Actor-Critic** algorithm from Sutton & Barto's "Reinforcement Learning: An Introduction" (Chapter 13).

## Algorithm

The One-Step Actor-Critic algorithm (episodic version):

```
Input: Differentiable policy π(a|s, θ) and state-value function V̂(s, w)
Parameters: Step sizes α^θ > 0, α^w > 0

Initialize policy parameter θ ∈ ℝ^d' and state-value weights w ∈ ℝ^d

Loop forever (for each episode):
    Initialize S (first state of episode)
    I ← 1
    
    Loop while S is not terminal (for each time step):
        A ~ π(·|S, θ)
        Take action A, observe S', R
        δ ← R + γV̂(S', w) - V̂(S, w)    [if S' is terminal, V̂(S', w) = 0]
        w ← w + α^w · δ · ∇V̂(S, w)
        θ ← θ + α^θ · I · δ · ∇ln π(A|S, θ)
        I ← γI
        S ← S'
```

## Project Structure

```
one_step_actor_critic/
├── __init__.py          # Package exports
├── agent.py             # Actor-Critic agent implementation
├── networks.py          # Neural network architectures (Actor, Critic)
├── environments.py      # Environment wrappers (CartPole, Blackjack)
├── config.py            # Configuration classes
├── utils.py             # Utility functions (logging, plotting, etc.)
├── experiment.py        # Experiment runner
├── train.py             # Training script (CLI)
└── requirements.txt     # Dependencies

report/
├── final_project_report.tex  # LaTeX report
└── final_project_report.aux
```

## Installation

```bash
# Create conda environment
conda create --name 687final python=3.10
conda activate 687final

# Install dependencies
cd one_step_actor_critic
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from one_step_actor_critic import (
    OneStepActorCritic,
    create_cartpole_env,
    preprocess_cartpole_state
)

# Create environment and agent
env = create_cartpole_env()
agent = OneStepActorCritic(
    state_dim=4,
    action_dim=2,
    actor_lr=1e-3,   # α^θ
    critic_lr=1e-3,  # α^w
    gamma=0.99       # γ
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    state = preprocess_cartpole_state(state)
    agent.reset_episode()  # Reset I ← 1
    
    done = False
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = preprocess_cartpole_state(next_state)
        
        agent.update(state, action, reward, next_state, done, log_prob)
        state = next_state
```

### Using the Experiment Runner

```python
from one_step_actor_critic import ExperimentRunner, TrainingConfig

config = TrainingConfig(
    env_name='cartpole',
    num_episodes=1000,
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    seed=42
)

runner = ExperimentRunner(config)
results = runner.run()
```

### Command Line

```bash
# Train on CartPole
python train.py --env cartpole --episodes 1000 --actor-lr 1e-3 --critic-lr 1e-3

# Train on Blackjack
python train.py --env blackjack --episodes 50000

# Use shared network architecture
python train.py --env cartpole --shared-network

# Save model and plots
python train.py --env cartpole --save-model model.pt --save-plot training.png
```

## Code Architecture

### Agent (`agent.py`)

Two implementations:

1. **`OneStepActorCritic`**: Separate actor and critic networks
   - Actor: π(a|s; θ) - policy network
   - Critic: V̂(s; w) - value network
   - Follows the pseudocode exactly

2. **`OneStepActorCriticShared`**: Shared feature extraction
   - Single network with two heads
   - More parameter-efficient

### Networks (`networks.py`)

- **`ActorNetwork`**: Outputs action probabilities (softmax)
- **`CriticNetwork`**: Outputs state value estimate
- **`ActorCriticNetwork`**: Combined network with shared layers

### Environments (`environments.py`)

- **CartPole-v1**: Classic control task
- **Blackjack-v1**: Card game environment

### Configuration (`config.py`)

```python
@dataclass
class TrainingConfig:
    env_name: str = 'cartpole'
    actor_lr: float = 1e-3      # α^θ
    critic_lr: float = 1e-3     # α^w
    gamma: float = 0.99         # γ
    hidden_dim: int = 128
    num_episodes: int = 1000
    ...
```

## Key Components Mapping to Pseudocode

| Pseudocode | Code |
|------------|------|
| θ (policy parameters) | `actor.parameters()` |
| w (value weights) | `critic.parameters()` |
| α^θ (policy step size) | `actor_lr` |
| α^w (value step size) | `critic_lr` |
| γ (discount factor) | `gamma` |
| I (discount for policy gradient) | `self.I` in agent |
| π(·\|S, θ) | `actor.forward()` |
| V̂(S, w) | `critic.forward()` |
| δ (TD error) | `td_error` in `update()` |

## Results

Run experiments and generate plots:

```bash
python experiment.py
```

Results are saved in the `results/` directory.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Chapter 13: Policy Gradient Methods