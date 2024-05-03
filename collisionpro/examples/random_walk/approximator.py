import copy
import numpy as np


class Approximator:
    def __init__(self, n_states, n_h, alpha=0.1, alpha_decrease=.9):
        self.n_states = n_states
        self.alpha = alpha
        self.alpha_decrease = alpha_decrease

        # Initialize table
        self.table = np.zeros((n_states, n_h))
        self.table[0, :] = -1.0
        self.table[-1, :] = -1.0

    def inference(self, states):
        return self.table[states]

    def fit(self, inputs, targets):
        inputs = inputs.astype(int)

        for idx in range(inputs.shape[0]):
            # Extract weighing from targets
            weighing = targets[idx, -1]

            # Update step w.r.t. alpha, weighing and TD-error
            self.table[inputs[idx]] = self.table[inputs[idx]] + self.alpha * weighing * (targets[idx, :-1] - self.table[inputs[idx]])

        self.alpha *= self.alpha_decrease

