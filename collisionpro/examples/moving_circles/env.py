from math import sqrt
import random
import numpy as np
from copy import deepcopy
import arcade


class HighLevelActions:
    UP = 0
    DOWN = 1
    NONE = 2

class Ego:
    def __init__(self, v_x, dt, radius, k_y=0.2, m=1):
        self.x = 0.0
        self.y = 0.0
        self.v_x = v_x
        self.v_y = 0.0

        self.radius = radius
        self.dt = dt
        self.m = m
        self.k_y = k_y
        self.y_0 = 0.0
        self.action = 0

        # Optimal damping constant for fast return
        self.d_y = 2 * sqrt(self.m * self.k_y)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, action_val):
        u = action_val

        self.x += self.v_x * self.dt

        force = u - self.k_y * (self.y - self.y_0) - self.d_y * self.v_y

        self.v_y += force * self.dt
        self.y += self.v_y * self.dt

    def __repr__(self):
        string = "Ego Vehicle\n"
        string += f"x      :: {self.x}\n"
        string += f"y      :: {self.y}\n"
        string += f"v_x    :: {self.v_x}\n"
        string += f"v_y    :: {self.v_y}\n"
        string += f"action :: {self.action}\n"
        return string


class Resonator:
    """
    The Resonator is an obstacle, that behaves like a spring-mass-system attached to the x-axis.
    """

    def __init__(self, x, y, v_y, m, k_y, k_x, r, u_max, dt, v_x=0.0):
        self.x = x
        self.x_0 = x
        self.y = y
        self.v_y = v_y
        self.v_x = v_x
        self.m = m
        self.k_y = k_y
        self.k_x = k_x
        self.r = r
        self.dt = dt
        self.a_y = 0.0
        self.u_max = u_max

        self.t_coll_shift = random.uniform(-1.5, 1.5)

        # Critical damping coefficient
        self.d = 2 * sqrt(self.m * self.k_x)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, ego):
        # Vertical update
        self.y += self.v_y * self.dt
        self.v_y += -(self.k_y / self.m) * self.y
        self.a_y = -(self.k_y / self.m) * self.y

        # Horizontal update

        # Movement towards ego trajectory
        if np.sign(self.y * self.v_y) < 0 and ego.x < self.x:

            res_time_to_traj_intersection = abs(self.y / self.v_y) if self.v_y != 0.0 else 1e3
            ego_time_to_traj_intersection = abs((self.x - ego.x) / ego.v_x) if ego.v_x != 0.0 else 1e3

            dist = np.linalg.norm([ego.x - self.x, ego.y - self.y])

            if res_time_to_traj_intersection < ego_time_to_traj_intersection + self.t_coll_shift:
                # Accelerate right
                u = min(1, (1 / dist) ** 1.5) * self.u_max

            else:
                # Accelerate left
                u = - min(1, (1 / dist) ** 1.5) * self.u_max

        else:
            u = 0.0

        force = u - self.k_x * (self.x - self.x_0) - self.d * self.v_x

        # Update horizontal dynamics
        self.v_x += force * self.dt
        self.x += self.v_x * self.dt

    def __repr__(self):
        string = "Resonator\n"
        string += f"y   :: {self.y}\n"
        string += f"x   :: {self.x}\n"
        string += f"x_0 :: {self.x_0}\n"
        string += f"v_y :: {self.v_y}\n"
        string += f"v_x :: {self.v_x}\n"
        string += f"a_y :: {self.a_y}\n"
        string += f"m   :: {self.m}\n"
        string += f"k_y   :: {self.k_y}\n"
        string += f"k_x :: {self.k_x}\n"
        string += f"r   :: {self.r}\n"
        string += f"d   :: {self.d}\n"
        string += f"u_max :: {self.u_max}\n"
        return string


class MovingCircles:
    def __init__(self,
                 radius_ego=1.5,
                 v_x_ego=1.0,
                 dt=0.1,
                 k_y=0.2,
                 m=1,
                 max_obstacles=1,
                 obstacle_creation_prop=0.01,
                 n_rel_obs=1,
                 state_type="compact",
                 action_type="continuous",  # continuous or discrete
                 noisy_perception=False):

        self.ego_init = Ego(v_x=v_x_ego, radius=radius_ego, dt=dt, k_y=k_y, m=m)

        self.n_res_dim = 6
        self.n_state_prefix = 3
        self.parking_prob = 0.0

        self.max_obstacles = max_obstacles
        self.obstacle_creation_prop = obstacle_creation_prop
        self.n_rel_obs = n_rel_obs
        self.dt = dt

        self.x_rear = -5
        self.x_front = 20
        self.y_height = 5

        self.high_level_actions = HighLevelActions

        if action_type in ["discrete", "continuous"]:
            self.action_type = action_type
        else:
            raise RuntimeError(f"Action type must be 'continuous' or 'discrete'. Got :: {action_type}")
        self.state_type = state_type  # "full" or "compact"
        self.noisy_perception = noisy_perception
        self.state_dim = self.n_state_prefix + self.n_res_dim * self.n_rel_obs

        self.ego = None
        self.collision = None
        self.obstacles = None
        self.relevant_obstacle = None
        self.state = None
        self.terminated = None
        self.action_val = None
        self.reset()

    def reset(self):
        self.ego = deepcopy(self.ego_init)
        self.collision = False
        self.obstacles = []
        self.relevant_obstacle = []
        self.terminated = False
        self.action_val = 0.0
        self.build_state()

    @staticmethod
    def get_ego_y(state):
        return state[0]

    @staticmethod
    def get_ego_v_x(state):
        return state[1]

    @staticmethod
    def get_ego_v_y(state):
        return state[2]

    def get_n_resonators(self, state):
        return round((state.size - self.n_state_prefix) / self.n_res_dim)

    def get_resonator_x(self, state, ith):
        return state[self.n_state_prefix + ith * self.n_res_dim + 0]

    def get_resonator_y(self, state, ith):
        return state[self.n_state_prefix + ith * self.n_res_dim + 1]

    def get_resonator_v_x(self, state, ith):
        return state[self.n_state_prefix + ith * self.n_res_dim + 2]

    def get_resonator_v_y(self, state, ith):
        return state[self.n_state_prefix + ith * self.n_res_dim + 3]

    def get_resonator_a_y(self, state, ith):
        return state[self.n_state_prefix + ith * self.n_res_dim + 4]

    def get_resonator_r(self, state, ith):
        return state[self.n_state_prefix + ith * self.n_res_dim + 5]

    def obstacle_is_relevant(self, obstacle):
        if self.ego.x + self.x_rear <= obstacle.x <= self.ego.x + self.x_front:
            return True
        return False

    def collision_detection(self):
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.ego.pos - obstacle.pos)
            if distance < obstacle.r + self.ego.radius:
                self.collision = True
                return

        self.collision = False

    def build_state(self):
        """
        TODO
        """

        # Add ego state
        state = [self.ego.y, self.ego.v_x, self.ego.v_y]

        # Reset relevant obstacles
        self.relevant_obstacle = []

        # Get next relevant obstacle
        n_rel_obs_ctr = 0
        for idx in range(len(self.obstacles)):
            if self.obstacles[idx].x + self.obstacles[idx].r > self.ego.x - 3.0:
                self.relevant_obstacle.append(self.obstacles[idx])
                n_rel_obs_ctr += 1

                if n_rel_obs_ctr == self.n_rel_obs:
                    break

        # Add each relevant obstacle to state
        for rel_obs in self.relevant_obstacle:
            state_obs = [rel_obs.x - self.ego.x,
                         rel_obs.y - self.ego.y,
                         rel_obs.v_x,
                         rel_obs.v_y,
                         rel_obs.a_y,
                         rel_obs.r]

            if self.noisy_perception:
                dist_to_obs = np.linalg.norm([rel_obs.x - self.ego.x, rel_obs.y - self.ego.y])
                state_obs[0] += np.random.normal(0, 0.001 * dist_to_obs)
                state_obs[1] += np.random.normal(0, 0.001 * dist_to_obs)
                state_obs[2] += np.random.normal(0, 0.0001 * dist_to_obs)
                state_obs[3] += np.random.normal(0, 0.0001 * dist_to_obs)
                state_obs[4] += np.random.normal(0, 0.0001 * dist_to_obs)
                state_obs[5] += np.random.normal(0, 0.001 * dist_to_obs)

            state = state + state_obs

        # If not enough relevant obstacles found, add dummies
        for idx in range(self.n_rel_obs - len(self.relevant_obstacle)):
            state.append(self.x_front)
            state.append(0)
            state.append(0)
            state.append(0)
            state.append(0)
            state.append(0)

        self.state = np.array(state)

    def get_action_value(self, action):
        # Get action value by action command
        if self.action_type == "discrete":
            if action == HighLevelActions.UP:
                action_val = 0.5
            elif action == HighLevelActions.DOWN:
                action_val = -0.5
            elif action == HighLevelActions.NONE:
                action_val = 0.0
            else:
                raise RuntimeError(f"Action type is invalid. Got :: {action}")
        else:  # continuous
            action_val = action

        return action_val

    def step(self, action):
        self.action_val = self.get_action_value(action)

        # Update ego
        self.ego.step(self.action_val)

        # Add new obstacles
        if len(self.obstacles) < self.max_obstacles and random.random() < self.obstacle_creation_prop:
            x = self.ego.x + self.x_front
            y = random.uniform(self.ego.radius + 2, self.y_height) * random.choice([-1, 1])
            v_y = 0
            r = float(max(0.1, np.random.normal(0.5, 0.1)))
            m = r ** 2 * 2
            k_y = random.uniform(0.0005, 0.005)
            k_x = k_y * float(np.clip(np.random.normal(5, 1), 0, 20))
            u_max = max(0.0, float(np.random.normal(2.0, 0.3)))

            if len(self.obstacles):
                if x - self.obstacles[-1].x > self.obstacles[-1].r + r:
                    self.obstacles.append(Resonator(x, y, v_y, m, k_y, k_x, r, u_max, self.dt))
            else:
                self.obstacles.append(Resonator(x, y, v_y, m, k_y, k_x, r, u_max, self.dt))

        # Update obstacles
        for obstacle in self.obstacles:
            if self.obstacle_is_relevant(obstacle):
                obstacle.step(self.ego)
            else:
                self.obstacles.remove(obstacle)

        # Collision detection
        self.collision_detection()

        # Rewards
        reward = -1.0 if self.collision else 0.0
        reward += -0.02 * abs(self.ego.y)

        self.terminated = True if self.collision else False
        info = {"collision": self.collision}

        # Build state
        self.build_state()

        return self.state, reward, self.terminated, False, info

    def rendering(self, controller, delta_time=None, max_length=None, EuclidToPixel=20):
        X_min = self.x_rear
        X_max = self.x_front
        X_width = X_max - X_min
        Y_min = -self.y_height
        Y_max = self.y_height
        Y_width = Y_max - Y_min
        P_width = X_width * EuclidToPixel
        P_height = Y_width * EuclidToPixel

        delta_time = self.dt if delta_time is None else delta_time

        ArcadeVisualization(P_width, P_height, self, controller, EuclidToPixel, delta_time)
        arcade.run()


# =================================================================================================
# -----   Rendering   -----------------------------------------------------------------------------
# =================================================================================================


class ArcadeVisualization(arcade.Window):
    def __init__(self, width, height, env, controller, EuclidToPixel=100, dt=.05):
        super().__init__(width, height, "Moving Circle Visualization")
        self.set_update_rate(dt)
        arcade.set_background_color(arcade.color.WHITE)
        self.env = env
        self.env.reset()
        self.controller = controller
        self.EuclidToPixel = EuclidToPixel
        self.X_min = env.x_rear
        self.X_max = env.x_front
        self.Y_min = -env.y_height
        self.X_width = env.x_front - env.x_rear
        self.Y_width = 2 * env.y_height
        self.state = None
        self.action = None
        self.action_val = 0.0

    def toPixel(self, x):
        return x * self.EuclidToPixel

    def toPixelCoord(self, x, y):
        return (x - self.X_min) * self.EuclidToPixel, (y - self.Y_min) * self.EuclidToPixel

    def on_draw(self):
        arcade.start_render()

        ego_x, ego_y = self.env.ego.pos
        ego_px, ego_py = self.toPixelCoord(0, ego_y)

        # Draw the vertical line
        v_line_ego = -ego_x % 1
        v_line_start = v_line_ego + self.X_min
        v_line_end = v_line_ego + self.X_max
        for x in np.linspace(v_line_start, v_line_end, int(self.X_width + 1)):
            px, _ = self.toPixelCoord(x, 0)
            arcade.draw_line(px, 0, px, self.toPixel(self.Y_width), (230, 230, 230), 1)

        # Visualize ego
        arcade.draw_circle_filled(ego_px, ego_py, self.toPixel(self.env.ego.radius), arcade.color.BLUE, num_segments=50)

        # Visualize obstacles
        for obs in self.env.obstacles:
            px, py = self.toPixelCoord(obs.x - ego_x, obs.y)
            arcade.draw_circle_filled(px, py, self.toPixel(obs.r), arcade.color.RED)

        # Visualize action
        if self.action_val > 0.0:
            arcade.draw_line(ego_px, ego_py, ego_px, ego_py + 50, (0, 200, 0), 4)
        elif self.action_val < 0.0:
            arcade.draw_line(ego_px, ego_py, ego_px, ego_py - 50, (0, 200, 0), 4)

        # Write q values
        px, py = self.toPixelCoord(self.X_min, self.Y_min)

    def update(self, delta_time):
        self.action = self.controller.get_action(self.env.state)
        self.action_val = self.env.get_action_value(self.action)
        self.state, _, terminated, truncated, _ = self.env.step(self.action)

        if terminated or truncated:
            self.close()


if __name__ == "__main__":
    class Controller:
        def __init__(self):
            self.q_values = [0, 0, 0]

        @staticmethod
        def get_action(state):
            return random.choice([HighLevelActions.UP, HighLevelActions.DOWN, HighLevelActions.NONE])

    env_ = MovingCircles()
    controller_ = Controller()
    env_.rendering(controller_, delta_time=0.025, EuclidToPixel=30)


