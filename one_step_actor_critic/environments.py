"""
Simple environments for Actor-Critic with function approximation.
"""

import gymnasium as gym
import numpy as np


class BlackjackEnv:
    
    def __init__(self):
        self.env = gym.make('Blackjack-v1', natural=False, sab=False)
        self.num_actions = 2
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
    
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.num_actions = 2
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


if __name__ == "__main__":
    print("Testing BlackjackEnv...")
    env = BlackjackEnv()
    state = env.reset()
    print(f"  num_actions: {env.num_actions}")
    print(f"  initial state: {state}")
    env.close()
    
    print("\nTesting CartPoleEnv...")
    env = CartPoleEnv()
    state = env.reset()
    print(f"  num_actions: {env.num_actions}")
    print(f"  initial state: {state}")
    env.close()
    
    print("\nAll environments tested!")