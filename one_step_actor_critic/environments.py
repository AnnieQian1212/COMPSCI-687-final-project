import gymnasium as gym
import numpy as np

class BlackjackEnv:
    def __init__(self):
        self._env = gym.make('Blackjack-v1', natural=False, sab=False)
        self.num_actions = 2
        self.state_dim = 3
        self.state_bounds = [(4, 21), (1, 10), (0, 1)]
        self.current_state = None

    def reset(self):
        state, _ = self._env.reset()
        self.current_state = np.array(state, dtype=float)
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        self.current_state = np.array(next_state, dtype=float)
        return self.current_state, reward, done

    def close(self):
        self._env.close()


class CartPoleEnv:
    def __init__(self):
        self._env = gym.make('CartPole-v1')
        self.num_actions = 2
        self.state_dim = 4
        self.state_bounds = [(-2.4, 2.4), (-3, 3), (-0.25, 0.25), (-3, 3)]
        self.current_state = None

    def reset(self):
        state, _ = self._env.reset()
        self.current_state = np.array(state, dtype=float)
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        self.current_state = np.array(next_state, dtype=float)
        return self.current_state, reward, done

    def close(self):
        self._env.close()

class FrozenLakeEnv:
    def __init__(self, slippery=False):
        self._env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=slippery)
        self.num_actions = 4
        self.state_dim = 2
        self.state_bounds = [(0, 3), (0, 3)]
        self.current_state = None

    def _state_to_coords(self, state):
        row = state // 4
        col = state % 4
        return np.array([row, col], dtype=float)

    def reset(self):
        state, _ = self._env.reset()
        self.current_state = self._state_to_coords(state)
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        self.current_state = self._state_to_coords(next_state)
        return self.current_state, reward, done

    def close(self):
        self._env.close()


class TaxiEnv:
    def __init__(self):
        self._env = gym.make('Taxi-v3')
        self.num_actions = 6
        self.state_dim = 4
        self.state_bounds = [(0, 4), (0, 4), (0, 4), (0, 3)]
        self.current_state = None

    def _decode_state(self, state):
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
        state, _ = self._env.reset()
        self.current_state = self._decode_state(state)
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        self.current_state = self._decode_state(next_state)
        return self.current_state, reward, done

    def close(self):
        self._env.close()


ENVS = {
    'blackjack': BlackjackEnv,
    'cartpole': CartPoleEnv,
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
