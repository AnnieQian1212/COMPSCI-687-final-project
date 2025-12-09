import numpy as np
from itertools import product


def softmax(x):
    x = np.clip(x, -500, 500)
    shifted = x - np.max(x)
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals) + 1e-8)


def normalize_state(state, bounds):
    state_arr = np.asarray(state, dtype=float)
    lows, highs = map(np.asarray, zip(*bounds))
    norm = (state_arr - lows) / (highs - lows + 1e-8)
    return np.clip(norm, 0.0, 1.0)

class TabularFeature:

    def __init__(self, num_states, state_bounds=None):
        self.num_states = num_states
        self.feature_dim = num_states
        print(f"Tabular feature: {num_states} states → {self.feature_dim} features")
    
    def __call__(self, state):

        if isinstance(state, (list, tuple, np.ndarray)):
            state = int(state[0])  
        else:
            state = int(state)
        

        one_hot = np.zeros(self.num_states)
        if 0 <= state < self.num_states:
            one_hot[state] = 1.0
        else:
            print(f"Warning: state {state} out of bounds [0, {self.num_states})")
        
        return one_hot


class PolynomialFeature:
    def __init__(self, state_dim, order=2, state_bounds=None):
        self.state_dim = state_dim
        self.order = order
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.power_terms = self._build_power_terms()
        self.feature_dim = len(self.power_terms)

    def _build_power_terms(self):
        terms = []
        for total_power in range(self.order + 1):
            for combo in product(range(total_power + 1), repeat=self.state_dim):
                if sum(combo) == total_power:
                    terms.append(combo)
        return terms

    def __call__(self, state):
        state_norm = normalize_state(state, self.state_bounds)
        return np.array([np.prod(state_norm ** powers) for powers in self.power_terms])


class FourierFeature:
    def __init__(self, state_dim, order=3, state_bounds=None):
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.coeff_grid = np.array(list(product(range(order + 1), repeat=state_dim)))
        self.feature_dim = len(self.coeff_grid)

    def __call__(self, state):
        state_norm = normalize_state(state, self.state_bounds)
        return np.cos(np.pi * (self.coeff_grid @ state_norm))


class TileCodingFeature:
    def __init__(self, state_dim, num_tilings=8, num_tiles=8, state_bounds=None):
        self.state_dim = state_dim
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.state_bounds = state_bounds or [(0, 1)] * state_dim

        self.offsets = np.random.rand(num_tilings, state_dim) / num_tiles
        self.tiles_per_tiling = num_tiles ** state_dim
        self.feature_dim = num_tilings * self.tiles_per_tiling

    def __call__(self, state):
        state_norm = normalize_state(state, self.state_bounds)
        features = np.zeros(self.feature_dim)

        for t_idx, offset in enumerate(self.offsets):
            shifted = np.clip((state_norm + offset) * self.num_tiles, 0, self.num_tiles - 1)
            tile_index = 0
            for value in shifted.astype(int):
                tile_index = tile_index * self.num_tiles + value
            features[t_idx * self.tiles_per_tiling + tile_index] = 1.0

        return features


class RBFFeature:
    def __init__(self, state_dim, centers_per_dim=6, sigma=0.2, state_bounds=None):
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.sigma = sigma

        axes = [np.linspace(0, 1, centers_per_dim) for _ in range(state_dim)]
        self.centers = np.array(list(product(*axes)))
        self.feature_dim = len(self.centers)

    def __call__(self, state):
        state_norm = normalize_state(state, self.state_bounds)
        diff = self.centers - state_norm
        squared_dist = np.sum(diff ** 2, axis=1)
        return np.exp(-squared_dist / (2 * self.sigma ** 2))


class CoarseCodingFeature:
    def __init__(self, state_dim, num_circles=50, radius=0.2, state_bounds=None):
        self.state_bounds = state_bounds or [(0, 1)] * state_dim
        self.radius = radius
        self.centers = np.random.rand(num_circles, state_dim)
        self.feature_dim = num_circles

    def __call__(self, state):
        state_norm = normalize_state(state, self.state_bounds)
        distances = np.linalg.norm(self.centers - state_norm, axis=1)
        return (distances <= self.radius).astype(float)


def build_feature(feature_type, state_dim, state_bounds, **kwargs):
    match feature_type:
        case "tabular":
            return TabularFeature(
                num_states=kwargs.get("num_states", 500),  # Default for Taxi
                state_bounds=state_bounds
            )
        case "poly":
            return PolynomialFeature(
                state_dim,
                order=kwargs.get("order", 2),
                state_bounds=state_bounds
            )
        case "fourier":
            return FourierFeature(
                state_dim,
                order=kwargs.get("order", 3),
                state_bounds=state_bounds
            )
        case "tile":
            return TileCodingFeature(
                state_dim,
                num_tilings=kwargs.get("num_tilings", 8),
                num_tiles=kwargs.get("num_tiles", 8),
                state_bounds=state_bounds
            )
        case "rbf":
            return RBFFeature(
                state_dim,
                centers_per_dim=kwargs.get("centers_per_dim", 6),
                sigma=kwargs.get("sigma", 0.2),
                state_bounds=state_bounds
            )
        case "coarse":
            return CoarseCodingFeature(
                state_dim,
                num_circles=kwargs.get("num_circles", 50),
                radius=kwargs.get("radius", 0.2),
                state_bounds=state_bounds
            )


class OneStepActorCritic:
    def __init__(self, num_actions, state_dim, alpha_theta, alpha_w, gamma,
                 feature_type="poly", state_bounds=None, **feature_kwargs):

        self.num_actions = num_actions
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma

        self.feature_fn = build_feature(
            feature_type, state_dim, state_bounds, **feature_kwargs
        )
        feat_dim = self.feature_fn.feature_dim

        print(f"Using {feature_type} features with dimension: {feat_dim}")

        self.policy_weights = np.zeros((num_actions, feat_dim))
        self.value_weights = np.zeros(feat_dim)
        self.return_coef = 1.0

    def reset_episode(self):
        self.return_coef = 1.0

    def get_value(self, state):
        features = self.feature_fn(state)
        return np.dot(self.value_weights, features)

    def get_pi(self, state):
        features = self.feature_fn(state)
        logits = self.policy_weights @ features
        return softmax(logits)

    def select_action(self, state):
        probs = self.get_pi(state)
        return np.random.choice(self.num_actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        features = self.feature_fn(state)

        next_value = 0.0 if done else self.get_value(next_state)
        curr_value = np.dot(self.value_weights, features)
        td_err = np.clip(reward + self.gamma * next_value - curr_value, -10, 10)

        self.value_weights += self.alpha_w * td_err * features

        probs = self.get_pi(state)
        grad_log_pi = -probs
        grad_log_pi[action] += 1.0

        self.policy_weights += (
            self.alpha_theta
            * self.return_coef
            * td_err
            * grad_log_pi[:, None]
            * features
        )

        self.return_coef *= self.gamma
        return td_err


def train(env, agent, num_episodes, verbose=True):
    reward_history = []
    td_error_history = []
    value_loss_history = []

    for ep in range(num_episodes):
        state = env.reset()
        agent.reset_episode()

        episode_reward = 0
        episode_td_list = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            td_err = agent.update(state, action, reward, next_state, done)

            episode_td_list.append(td_err)
            episode_reward += reward
            state = next_state

        reward_history.append(episode_reward)
        td_error_history.append(np.mean(episode_td_list))

        if verbose and (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}, Avg Reward: {np.mean(reward_history[-100:]):.2f}")

    return {
        "rewards": reward_history,
        "value_losses": value_loss_history,
        "td_errors": td_error_history
    }
