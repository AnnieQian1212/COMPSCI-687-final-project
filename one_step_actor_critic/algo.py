"""
One-Step Actor-Critic with Multiple Feature Approximations
Based on Sutton & Barto Chapter 13
"""

import numpy as np
from itertools import product


def softmax(x):
    x = np.clip(x, -500, 500)
    t = np.exp(x - np.max(x))
    return t / (np.sum(t) + 1e-8)


def normalize_state(state, bounds):
    """Normalize state to [0, 1]"""
    state = np.atleast_1d(state).astype(float)
    normalized = np.zeros(len(bounds))
    for i, (low, high) in enumerate(bounds):
        normalized[i] = (state[i] - low) / (high - low + 1e-8)
    return np.clip(normalized, 0, 1)


# ==================== Feature Classes ====================

class PolynomialFeature:
    """Polynomial basis: x(s) = [1, s1, s2, ..., s1^2, s1*s2, ...]"""
    
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
    
    def __call__(self, state):
        s = normalize_state(state, self.state_bounds)
        features = np.zeros(self.feature_dim)
        for i, p in enumerate(self.powers):
            features[i] = np.prod([s[j] ** p[j] for j in range(self.state_dim)])
        return features


class FourierFeature:
    """Fourier basis: φ_c(s) = cos(π * c · s)"""
    
    def __init__(self, state_dim, order=3, state_bounds=None):
        self.state_dim = state_dim
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.coeffs = np.array(list(product(range(order + 1), repeat=state_dim)))
        self.feature_dim = len(self.coeffs)
    
    def __call__(self, state):
        s = normalize_state(state, self.state_bounds)
        return np.cos(np.pi * self.coeffs.dot(s))


class TileCodingFeature:
    """Tile Coding with multiple tilings"""
    
    def __init__(self, state_dim, num_tilings=8, num_tiles=8, state_bounds=None):
        self.state_dim = state_dim
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.feature_dim = num_tilings * (num_tiles ** state_dim)
        self.offsets = np.random.rand(num_tilings, state_dim) / num_tiles
    
    def __call__(self, state):
        s = normalize_state(state, self.state_bounds)
        features = np.zeros(self.feature_dim)
        tiles_per_tiling = self.num_tiles ** self.state_dim
        
        for tiling in range(self.num_tilings):
            shifted = s + self.offsets[tiling]
            tile_indices = np.clip((shifted * self.num_tiles).astype(int), 0, self.num_tiles - 1)
            
            flat_idx = 0
            for d in range(self.state_dim):
                flat_idx = flat_idx * self.num_tiles + tile_indices[d]
            
            features[tiling * tiles_per_tiling + flat_idx] = 1.0
        
        return features


class RBFFeature:
    """Radial Basis Function with grid centers"""
    
    def __init__(self, state_dim, centers_per_dim=6, sigma=0.2, state_bounds=None):
        self.state_dim = state_dim
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.sigma = sigma
        
        grid_axes = [np.linspace(0, 1, centers_per_dim) for _ in range(state_dim)]
        self.centers = np.array(list(product(*grid_axes)))
        self.feature_dim = len(self.centers)
    
    def __call__(self, state):
        s = normalize_state(state, self.state_bounds)
        diff = self.centers - s
        squared_dist = np.sum(diff ** 2, axis=1)
        return np.exp(-squared_dist / (2 * self.sigma ** 2))


class CoarseCodingFeature:
    """Coarse Coding with random circles"""
    
    def __init__(self, state_dim, num_circles=50, radius=0.2, state_bounds=None):
        self.state_dim = state_dim
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.radius = radius
        self.centers = np.random.rand(num_circles, state_dim)
        self.feature_dim = num_circles
    
    def __call__(self, state):
        s = normalize_state(state, self.state_bounds)
        features = np.zeros(self.feature_dim)
        for i, center in enumerate(self.centers):
            dist = np.linalg.norm(s - center)
            if dist <= self.radius:
                features[i] = 1.0
        return features


# ==================== Feature Factory ====================

def build_feature(feature_type, state_dim, state_bounds, **kwargs):
    """Factory function to build feature extractor"""
    
    if feature_type == "poly":
        order = kwargs.get("order", 2)
        return PolynomialFeature(state_dim, order=order, state_bounds=state_bounds)
    
    elif feature_type == "fourier":
        order = kwargs.get("order", 3)
        return FourierFeature(state_dim, order=order, state_bounds=state_bounds)
    
    elif feature_type == "tile":
        num_tilings = kwargs.get("num_tilings", 8)
        num_tiles = kwargs.get("num_tiles", 8)
        return TileCodingFeature(state_dim, num_tilings=num_tilings, num_tiles=num_tiles, state_bounds=state_bounds)
    
    elif feature_type == "rbf":
        centers_per_dim = kwargs.get("centers_per_dim", 6)
        sigma = kwargs.get("sigma", 0.2)
        return RBFFeature(state_dim, centers_per_dim=centers_per_dim, sigma=sigma, state_bounds=state_bounds)
    
    elif feature_type == "coarse":
        num_circles = kwargs.get("num_circles", 50)
        radius = kwargs.get("radius", 0.2)
        return CoarseCodingFeature(state_dim, num_circles=num_circles, radius=radius, state_bounds=state_bounds)
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


# ==================== Actor-Critic Agent ====================

class OneStepActorCritic:
    """One-step Actor-Critic with linear function approximation"""
    
    def __init__(self, num_actions, state_dim, alpha_theta, alpha_w, gamma, 
                 feature_type="poly", state_bounds=None, **feature_kwargs):
        self.num_actions = num_actions
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        
        # Extract feature-specific kwargs
        self.feature = build_feature(feature_type, state_dim, state_bounds, **feature_kwargs)
        feature_dim = self.feature.feature_dim
        
        print(f"Using {feature_type} features with dimension: {feature_dim}")
        
        self.theta = np.zeros((num_actions, feature_dim))
        self.w = np.zeros(feature_dim)
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
        
        v_next = 0.0 if done else self.get_value(next_state)
        v_curr = np.dot(self.w, x)
        delta = reward + self.gamma * v_next - v_curr
        
        # value_loss = delta ** 2
        delta = np.clip(delta, -10, 10)
        
        self.w += self.alpha_w * delta * x
        
        pi = self.get_pi(state)
        grad_log_pi = -pi.copy()
        grad_log_pi[action] += 1.0
        
        for a in range(self.num_actions):
            self.theta[a] += self.alpha_theta * self.I * delta * grad_log_pi[a] * x
        
        self.I *= self.gamma
        
        return delta


# ==================== Training Function ====================

def train(env, agent, num_episodes, verbose=True):
    rewards_history = []
    value_losses_history = []
    td_errors_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()
        episode_reward = 0
        # episode_value_loss = []
        episode_td_errors = []
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            delta = agent.update(state, action, reward, next_state, done)
            
            # episode_value_loss.append(value_loss)
            episode_td_errors.append(delta)
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        # value_losses_history.append(np.mean(episode_value_loss))
        td_errors_history.append(np.mean(episode_td_errors))
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            # avg_loss = np.mean(value_losses_history[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    return {
        'rewards': rewards_history,
        'value_losses': value_losses_history,
        'td_errors': td_errors_history
    }