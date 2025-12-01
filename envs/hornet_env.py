import gym
from gym import spaces
import numpy as np

class KnightHornetEnv(gym.Env):
    """
    Original RL environment: Knight vs Hornet duel simulator.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # ----- Action space -----
        self.action_space = spaces.Discrete(8)  
        # 0 idle, 1 dash, 2 Lattack, 3 Rattack, 4 Uattack, 5 left, 6 right, 7 jump

        # ----- Observation space -----
        # [knight_x, knight_y, knight_vx, knight_vy,
        #  hornet_x, hornet_y, hornet_state, hornet_timer,
        #  knight_state, knight_timer]
        high = np.array([1000]*10, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ----- Game parameters -----
        self.gravity = -0.4
        self.knight_speed = 1.0
        self.knight_jump_speed = 8.0
        self.knight_dash_speed = 3.0
        self.knight_attack_range = 1.2

        # Hornet behavior probabilities
        self.hornet_probs = {
            "idle": 0.4,
            "dash": 0.3,
            "jump": 0.2,
            "stab": 0.1
        }

        self.reset()

    def reset(self):
        # Knight
        self.knight_x = 0.0
        self.knight_y = 0.0
        self.knight_vx = 0.0
        self.knight_vy = 0.0
        self.knight_state = 0        # 0 idle, 1 attack, 2 dash, 3 hurt
        self.knight_timer = 0

        # Hornet
        self.hornet_x = 3.0
        self.hornet_y = 0.0
        self.hornet_state = 0        # same meaning
        self.hornet_timer = 0

        self.t = 0

        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.knight_x, self.knight_y,
            self.knight_vx, self.knight_vy,
            self.hornet_x, self.hornet_y,
            self.hornet_state, self.hornet_timer,
            self.knight_state, self.knight_timer,
        ], dtype=np.float32)

    # ------------------------ Hornet AI ------------------------
    def _update_hornet_ai(self):
        if self.hornet_timer > 0:
            self.hornet_timer -= 1
            return

        # Select new action
        r = np.random.rand()
        acc = 0
        for act, p in self.hornet_probs.items():
            acc += p
            if r <= acc:
                chosen = act
                break

        if chosen == "idle":
            self.hornet_state = 0
            self.hornet_timer = 20
        elif chosen == "dash":
            self.hornet_state = 1
            self.hornet_timer = 15
        elif chosen == "jump":
            self.hornet_state = 2
            self.hornet_timer = 25
        elif chosen == "stab":
            self.hornet_state = 3
            self.hornet_timer = 18

    # ------------------------ Step ------------------------
    def step(self, action):
        reward = 0

        # --- Knight action ---
        if self.knight_timer > 0:
            self.knight_timer -= 1
        else:
            if action == 1:
                self.knight_vx = -self.knight_speed
            elif action == 2:
                self.knight_vx = self.knight_speed
            elif action == 3:
                if self.knight_y == 0:
                    self.knight_vy = self.knight_jump_speed
            elif action == 4:
                self.knight_state = 1
                self.knight_timer = 10
                # Check attack hit
                if abs(self.knight_x - self.hornet_x) < self.knight_attack_range:
                    reward += 10
            elif action == 5:
                self.knight_state = 2
                self.knight_timer = 8
                self.knight_vx = np.sign(self.hornet_x - self.knight_x) * self.knight_dash_speed

        # --- Physics ---
        self.knight_x += self.knight_vx
        self.knight_y += self.knight_vy
        self.knight_vy += self.gravity

        if self.knight_y < 0:
            self.knight_y = 0
            self.knight_vy = 0

        # --- Hornet ---
        self._update_hornet_ai()

        # Hornet dash hitbox
        if self.hornet_state == 1:
            if abs(self.knight_x - self.hornet_x) < 0.8:
                reward -= 10

        # --- End condition ---
        done = False
        if reward >= 50:
            done = True

        self.t += 1
        if self.t >= 2000:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"[t={self.t}] Knight({self.knight_x:.2f}) Hornet({self.hornet_x:.2f})")
