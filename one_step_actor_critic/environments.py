"""
Simple environments for Actor-Critic with function approximation.
"""

import gymnasium as gym
import numpy as np


class BlackjackEnv:
    """Blackjack: state_dim=3, num_actions=2"""
    
    def __init__(self):
        self.env = gym.make('Blackjack-v1', natural=False, sab=False)
        self.num_actions = 2
        self.state_dim = 3
        self.state_bounds = [(4, 21), (1, 10), (0, 1)]
        self.state = None
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = np.array(state, dtype=float)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = np.array(next_state, dtype=float)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class CartPoleEnv:
    """CartPole: state_dim=4, num_actions=2"""
    
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.num_actions = 2
        self.state_dim = 4
        self.state_bounds = [(-2.4, 2.4), (-3, 3), (-0.25, 0.25), (-3, 3)]
        self.state = None
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = np.array(state, dtype=float)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = np.array(next_state, dtype=float)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class MountainCarEnv:
    """MountainCar: state_dim=2, num_actions=3"""
    
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.num_actions = 3
        self.state_dim = 2
        self.state_bounds = [(-1.2, 0.6), (-0.07, 0.07)]
        self.state = None
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = np.array(state, dtype=float)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = np.array(next_state, dtype=float)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class AcrobotEnv:
    """Acrobot: state_dim=6, num_actions=3"""
    
    def __init__(self):
        self.env = gym.make('Acrobot-v1')
        self.num_actions = 3
        self.state_dim = 6
        self.state_bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-12, 12), (-28, 28)]
        self.state = None
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = np.array(state, dtype=float)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = np.array(next_state, dtype=float)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class LunarLanderEnv:
    """LunarLander: state_dim=8, num_actions=4"""
    
    def __init__(self):
        self.env = gym.make('LunarLander-v3')
        self.num_actions = 4
        self.state_dim = 8
        self.state_bounds = [
            (-1.5, 1.5),   # x
            (-1.5, 1.5),   # y
            (-5, 5),       # vx
            (-5, 5),       # vy
            (-3.14, 3.14), # angle
            (-5, 5),       # angular velocity
            (0, 1),        # left leg contact
            (0, 1)         # right leg contact
        ]
        self.state = None
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = np.array(state, dtype=float)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = np.array(next_state, dtype=float)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class CliffWalkingEnv:
    """CliffWalking: state_dim=1 (tabular), num_actions=4"""
    
    def __init__(self):
        self.env = gym.make('CliffWalking-v1')
        self.num_actions = 4
        self.state_dim = 2
        self.state_bounds = [(0, 3), (0, 11)]  # row, col
        self.state = None
    
    def _to_2d(self, state):
        row = state // 12
        col = state % 12
        return np.array([row, col], dtype=float)
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = self._to_2d(state)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = self._to_2d(next_state)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class FrozenLakeEnv:
    """FrozenLake: state_dim=2, num_actions=4"""
    
    def __init__(self, is_slippery=False):
        self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
        self.num_actions = 4
        self.state_dim = 2
        self.state_bounds = [(0, 3), (0, 3)]
        self.state = None
    
    def _to_2d(self, state):
        row = state // 4
        col = state % 4
        return np.array([row, col], dtype=float)
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = self._to_2d(state)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = self._to_2d(next_state)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


class TaxiEnv:
    """Taxi: state_dim=4, num_actions=6"""
    
    def __init__(self):
        self.env = gym.make('Taxi-v3')
        self.num_actions = 6
        self.state_dim = 4
        self.state_bounds = [(0, 4), (0, 4), (0, 4), (0, 3)]
        self.state = None
    
    def _decode(self, state):
        # taxi_row, taxi_col, passenger_loc, destination
        out = []
        out.append(state % 5)
        state //= 5
        out.append(state % 5)
        state //= 5
        out.append(state % 5)
        state //= 5
        out.append(state)
        return np.array(out[::-1], dtype=float)
    
    def reset(self):
        state, _ = self.env.reset()
        self.state = self._decode(state)
        return self.state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.state = self._decode(next_state)
        return self.state, reward, done
    
    def close(self):
        self.env.close()


ENVS = {
    'blackjack': BlackjackEnv,
    'cartpole': CartPoleEnv,
    'mountaincar': MountainCarEnv,
    'acrobot': AcrobotEnv,
    'lunarlander': LunarLanderEnv,
    'cliffwalking': CliffWalkingEnv,
    'frozenlake': FrozenLakeEnv,
    'taxi': TaxiEnv,
}


def make_env(name):

    if name not in ENVS:
        raise ValueError(f"Unknown env: {name}. Available: {list(ENVS.keys())}")
    return ENVS[name]()


if __name__ == "__main__":
    for name, EnvClass in ENVS.items():
        print(f"Testing {name}...")
        env = EnvClass()
        state = env.reset()
        print(f"  state_dim: {env.state_dim}, num_actions: {env.num_actions}")
        print(f"  state_bounds: {env.state_bounds}")
        print(f"  initial state: {state}")
        
        action = np.random.randint(env.num_actions)
        next_state, reward, done = env.step(action)
        print(f"  after action {action}: state={next_state}, reward={reward}, done={done}")
        env.close()
        print()
    
    print("All environments tested!")