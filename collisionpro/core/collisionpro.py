import copy
import numpy as np
import random
from typing import List, Callable


class Sample:
    def __init__(self, states, steps2collision, rewards, action=None):
        self.action = action
        self.states = states
        self.rewards = rewards
        self.steps2collision = steps2collision
        self.indices = None

    @property
    def cur_state(self):
        return self.states[0]

    @property
    def td_states(self):
        return self.states[1:]

    def is_collision_predecessor(self):
        if self.steps2collision == -1:
            return False
        return True


class CollisionPro:
    def __init__(self,
                 env,
                 n_h,
                 p_c,
                 p_nc,
                 lambda_val,
                 n_stacking=1,
                 td_max=None,
                 controller=None,
                 ):
        """
        :param p_c: Sampling probability for collision samples
        :param p_nc: Sampling probability for non-collision samples
        :param lambda_val: Lambda value for TD(lambda)
        :param n_h: Number of heads/predictors
        :param td_max: Maximum return horizon (td_max <= N_h)
        """

        self.env = env
        self.controller = controller

        # Parameters
        self.p_c = p_c
        self.p_nc = p_nc
        self.lambda_val = lambda_val
        self.td_max = n_h if td_max is None else td_max
        self.n_h = n_h

        self.n_stacking = n_stacking

        self.lambda_mat = self._get_lambda_matrix()
        self.eval_samples = None
        self.eval_hist = []

    # =========================================================================
    # --- Sample Generation Functionality -------------------------------------
    # =========================================================================

    def run_episode(self):
        self.env.reset()
        new_state = self.env.state

        episode = {
            "observations": [],
            "collision": False,
            "actions": [],
            "rewards": []
        }

        # Run episode
        reward = 0
        while True:
            action = None
            if self.controller:
                action = self.controller.get_action(self.env.state)

            episode["observations"].append(new_state)
            episode["actions"].append(action)
            episode["rewards"].append(reward)

            new_state, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                episode["observations"].append(new_state)
                episode["actions"].append(None)
                episode["rewards"].append(reward)
                break

        episode["collision"] = True if info["collision"] else False

        return episode

    def stacking(self, episode):
        samples_stacked = []
        obs_stacked = []
        for idx in range(len(episode["observations"])):
            obs_stacked.append(episode["observations"][idx])
            obs_stacked = obs_stacked[-self.n_stacking:]
            if len(obs_stacked) == self.n_stacking:
                samples_stacked.append({
                    "state": np.stack(obs_stacked),
                    "reward": episode["rewards"][idx],
                    "action": episode["actions"][idx],
                })
        return {"samples": samples_stacked, "collision": episode["collision"]}

    def create_td_samples(self, episode):
        episode_len = len(episode["samples"])

        # Calculate number of steps to collision (-1 means collision event isn't relevant)
        steps_to_collision = [-1] * episode_len
        if episode["collision"]:
            for idx in range(min(self.td_max + 1, len(steps_to_collision))):
                steps_to_collision[-idx - 1] = idx

        # Create samples
        td_samples = []
        states = [ele["state"] for ele in episode["samples"]]
        rewards = [ele["reward"] for ele in episode["samples"]]
        for idx in range(episode_len):
            td_samples.append(Sample(states=states[idx:idx + self.td_max + 1],
                                  steps2collision=steps_to_collision[idx],
                                  rewards=rewards[idx:idx + self.td_max + 1]))

        return {"td_samples": td_samples, "collision": episode["collision"]}

    def sampling(self, episode):
        if episode["collision"]:
            episode_nc = episode["td_samples"][:-self.n_h - 1]
            episode_c = episode["td_samples"][-self.n_h - 1:]
        else:
            episode_nc = episode["td_samples"]
            episode_c = []

        samples_nc = random.sample(episode_nc, round(len(episode_nc) * self.p_nc))
        samples_c = random.sample(episode_c, round(len(episode_c) * self.p_c))

        return samples_nc + samples_c

    def generate_samples(self, n, return_all=False):
        samples = []
        all_samples = []
        info = {
            "rewards": []
        }

        while len(samples) < n:
            episode = self.run_episode()
            episode = self.stacking(episode)
            episode = self.create_td_samples(episode)
            new_samples = self.sampling(episode)
            samples.extend(new_samples)
            if return_all:
                all_samples.extend(episode)

        if return_all:
            return random.sample(samples, n), all_samples
        else:
            return random.sample(samples, n)

    # =========================================================================
    # --- Target Generation Functionality -------------------------------------
    # =========================================================================

    def _get_lambda_matrix(self):
        # Generate lambda vector
        lambda_vec = (1 - self.lambda_val) * np.array([self.lambda_val ** (power + 1) for power in range(self.td_max)])

        # Replicate for all rows
        lambda_mat = np.tile(lambda_vec, (self.n_h, 1))

        # Set values that are unreachable for the heads to zero.
        for idx_r in range(self.n_h):
            for idx_c in range(self.td_max):
                if idx_r < idx_c:
                    lambda_mat[idx_r, idx_c] = 0.0

        # Add remainder to the last element (Sum must be 1)
        for idx_r in range(self.n_h):
            idx_c = min(idx_r, self.td_max - 1)
            lambda_mat[idx_r, idx_c] += (1 - np.sum(lambda_mat[idx_r, :]))

        return lambda_mat

    def generate_evaluation_samples(self, N_samp_eval, p_s=0.1):
        """
        :param N_samp_eval: Number of evaluation samples
        :param p_s: Probability of taking a sample
        :return: A dict with a list of states and their targets
        """

        eval_samples = {
            "inputs": [],
            "targets": [],
        }

        while len(eval_samples["inputs"]) < N_samp_eval:
            episode = self.run_episode()
            episode = self.stacking(episode)
            episode = self.create_td_samples(episode)

            N_transitions = len(episode) - 1
            N_samples = int(N_transitions * p_s)

            random_indices = np.random.randint(0, N_transitions, N_samples)
            random_samples = [episode[i] for i in random_indices]

            steps_to_collision = N_transitions - np.array(random_indices)
            steps_to_collision[steps_to_collision > self.n_h] = self.n_h + 1

            target_values = np.zeros((len(random_samples), self.n_h), dtype=float)

            for idx, steps in enumerate(steps_to_collision):
                target_values[idx, :] = np.concatenate([np.zeros(steps - 1, dtype=float),
                                                        -np.ones(self.n_h - steps + 1, dtype=float)])

            for idx in range(len(random_samples)):
                eval_samples["inputs"].append(random_samples[idx].cur_state)
                eval_samples["targets"].append(target_values[idx, :])

        return eval_samples

    def set_evaluation_samples(self, eval_samples):
        self.eval_samples = eval_samples

    def evaluate(self, func_inference, verbose=True, eval_samples=None):
        if eval_samples is None:
            cur_eval_samples = self.eval_samples
        else:
            cur_eval_samples = eval_samples

        pred = func_inference(np.row_stack(cur_eval_samples["inputs"]))
        targets = np.row_stack(cur_eval_samples["targets"])

        errors_acc = np.mean(np.square(pred - targets), axis=0)
        errors_pes = np.mean(pred - targets, axis=0)

        if verbose:
            print(f"Mean acc. error :: {np.mean(errors_acc):.2e} | Mean pes. error :: {np.mean(errors_pes):.2e}")

        self.eval_hist.append({"acc": errors_acc, "pes": errors_pes})
        return errors_acc, errors_pes

    def generate_training_data(self, samples: List[Sample], func_inference: Callable):

        input_dim = samples[0].cur_state.shape
        n_samples = len(samples)

        # =================================================
        # --- Generate inputs -----------------------------
        # =================================================

        inputs = np.zeros((n_samples, *input_dim)).squeeze()

        for idx, sample in enumerate(samples):
            inputs[idx] = sample.cur_state

        # =================================================
        # --- Evaluate future states ----------------------
        # =================================================

        future_states = []

        idx_global = 0

        for sample in samples:
            future_states.extend(sample.td_states)
            n_future_states = len(sample.td_states)
            sample.indices = (list(range(idx_global, idx_global + n_future_states)))
            idx_global += n_future_states

        future_states = np.row_stack(future_states).squeeze()

        P_pred = func_inference(future_states)

        # =================================================
        # --- Generate targets ----------------------------
        # =================================================

        targets = np.zeros((n_samples, self.n_h + 1))
        P_1_to_n_tilde_template = np.tril(-np.ones((self.n_h, self.td_max), dtype=float))

        for idx, sample in enumerate(samples):
            P_1_to_n_tilde = copy.deepcopy(P_1_to_n_tilde_template)

            if sample.is_collision_predecessor():
                td_steps = min(len(sample.td_states) - 1, sample.steps2collision - 1)
                for idx_td in range(len(sample.td_states) - 1):
                    V_k = P_pred[sample.indices[idx_td]]
                    P_1_to_n_tilde[:, idx_td] = np.concatenate([np.zeros(idx_td + 1), V_k[:-(idx_td + 1)]])
            else:
                for idx_td in range(self.td_max):
                    V_k = P_pred[sample.indices[idx_td]]
                    P_1_to_n_tilde[:, idx_td] = np.concatenate([np.zeros(idx_td + 1), V_k[:-(idx_td + 1)]])

            # Calculate TD-errors and loss weights
            targets[idx, :-1] = np.squeeze(np.sum(P_1_to_n_tilde * self.lambda_mat, axis=1))
            targets[idx, -1] = self.p_nc / self.p_c if sample.is_collision_predecessor() else 1.0

        return inputs, targets









