"""
One-Step Actor-Critic with Polynomial Feature Approximation
Based on Sutton & Barto Chapter 13
"""

import numpy as np
from itertools import product


def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)


class PolynomialFeature:
    """x(s) = [1, s1, s2, ..., s1^2, s1*s2, ...]"""
    
    def __init__(self, state_dim, order=2, state_bounds=None):
        self.state_dim = state_dim
        self.order = order
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.powers = self._generate_powers()
        self.feature_dim = len(self.powers)
    
    def _generate_powers(self):
        powers = []
        for o in range(self.order + 1):
            for p in product(range(o + 1), repeat=self.state_dim):
                if sum(p) == o:
                    powers.append(p)
        return powers
    
    def _normalize(self, state):
        state = np.atleast_1d(state).astype(float)
        normalized = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            low, high = self.state_bounds[i]
            normalized[i] = (state[i] - low) / (high - low + 1e-8)
        return np.clip(normalized, 0, 1)
    
    def __call__(self, state):
        s = self._normalize(state)
        features = np.zeros(self.feature_dim)
        for i, p in enumerate(self.powers):
            features[i] = np.prod([s[j] ** p[j] for j in range(self.state_dim)])
        return features


class OneStepActorCritic:
    """
    One-step Actor-Critic with polynomial function approximation.
    
    h(s,a,θ) = θ_a^T x(s)
    π(a|s,θ) = softmax(h(s,:,θ))
    V(s,w) = w^T x(s)
    """
    
    def __init__(self, num_actions, state_dim, alpha_theta, alpha_w, gamma, 
                 order=2, state_bounds=None):
        self.num_actions = num_actions
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        
        self.feature = PolynomialFeature(state_dim, order, state_bounds)
        feature_dim = self.feature.feature_dim
        
        # θ[a, :] - policy parameters for each action
        self.theta = np.zeros((num_actions, feature_dim))
        # w - state value weights
        self.w = np.zeros(feature_dim)
        # I - discount factor for policy gradient
        self.I = 1.0
    
    def reset_episode(self):
        self.I = 1.0
    
    def get_value(self, state):
        x = self.feature(state)
        return np.dot(self.w, x)
    
    def get_pi(self, state):
        x = self.feature(state)
        h = np.dot(self.theta, x)
        return softmax(h)
    
    def select_action(self, state):
        pi = self.get_pi(state)
        return np.random.choice(self.num_actions, p=pi)
    
    def update(self, state, action, reward, next_state, done):
        x = self.feature(state)
        
        # δ ← R + γ * V(S') - V(S)
        v_next = 0.0 if done else self.get_value(next_state)
        v_curr = np.dot(self.w, x)
        delta = reward + self.gamma * v_next - v_curr
        
        # w ← w + α^w * δ * x(S)
        self.w += self.alpha_w * delta * x
        
        # ∇ln π(a|s) for softmax: x(s) * (e_a - π)
        pi = self.get_pi(state)
        grad_log_pi = -pi.copy()
        grad_log_pi[action] += 1.0
        
        # θ_a ← θ_a + α^θ * I * δ * grad_log_pi[a] * x
        for a in range(self.num_actions):
            self.theta[a] += self.alpha_theta * self.I * delta * grad_log_pi[a] * x
        
        # I ← γ * I
        self.I *= self.gamma
        
        return delta


def train(env, agent, num_episodes, verbose=True):
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg:.2f}")
    
    return rewards_history