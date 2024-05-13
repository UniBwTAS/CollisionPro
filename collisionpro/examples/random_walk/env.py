import random
import numpy as np


class RandomWalk:
    """
    Random Walk consists of N consecutive states. The first and last state are terminal states (collision states).
    The probability of going right or left is 0.5.
    The reward is always 0, except for a collision where it is -1.
    """

    RIGHT = 0
    LEFT = 1

    def __init__(self, n_states=9):
        if n_states < 3:
            raise RuntimeError("Number of states must be at least 3!")

        self.n_states = n_states
        self.end_states = [0, n_states - 1]

        self.state = None
        self.terminated = None
        self.reset()

    def reset(self):
        self.state = np.array([np.random.randint(1, self.n_states - 1)])
        self.terminated = False

    def is_terminated(self):
        if self.state in self.end_states:
            return True
        return False

    def step(self, action):
        action = random.randint(0, 1)
        self.state = self.state + 1 if action == RandomWalk.RIGHT else self.state - 1
        self.terminated = self.is_terminated()
        reward = -1 if self.terminated else 0

        return self.state, reward, self.terminated, False, {"collision": self.terminated}
