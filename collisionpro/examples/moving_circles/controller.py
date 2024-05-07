import numpy as np

from collisionpro.examples.moving_circles.env import MovingCircles


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0., theta=0.15, sigma=0.05):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return np.copy(self.state)


class Controller:
    def __init__(self, env, add_noise=False, mu=0., sigma=0., theta=0., u_y_max=48, u_repulsive=1.):
        self.add_noise = add_noise
        self.noise_generator = OrnsteinUhlenbeckNoise(1, mu, sigma, theta)
        self.u_y_max = u_y_max
        self.u_repulsive = u_repulsive
        self.env: MovingCircles = env

    def get_action(self, state):
        if self.add_noise:
            noise = self.noise_generator.sample()[0]
        else:
            noise = 0.0

        obs_rel = None
        for obs in self.env.obstacles:
            if self.env.obstacle_is_relevant(obs):
                obs_rel = obs
                break

        if obs_rel is None:
            return 0.0

        # Get distance to next resonator
        x_rel = self.env.ego.x - obs_rel.x
        dist = np.linalg.norm([self.env.ego.x - obs_rel.x, self.env.ego.y - obs_rel.y])

        # Get y-dist between ego and resonator
        y_dist = obs_rel.y - self.env.ego.y

        # Based on distance and y-offset calculate dodge maneuver
        u = np.clip(-np.sign(y_dist) * (1 / dist) ** 1.5 * self.u_repulsive + noise, -self.u_y_max, self.u_y_max)

        return u