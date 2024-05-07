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
    def target_states(self):
        return self.states[1:]

    def is_collision_predecessor(self):
        if self.steps2collision == -1:
            return False
        return True


class CollisionPro:
    def __init__(self,
                 env,
                 n_h,
                 p_c=1.0,
                 p_nc=1.0,
                 lambda_val=0.5,
                 n_stacking=1,
                 td_max=None,
                 controller=None,
                 ):
        """
        CollisionPro is a Python library designed for sample generation, target generation, and evaluation
        in collision avoidance tasks. It provides functionalities for:

        - Generating collision and non-collision samples from a given environment based on specified sampling strategy (p_c, p_nc).
        - Calculating fixed-finite TD-lambda targets for collision events.
        - Evaluating the performance of a collision avoidance algorithm.

        See the documentation for more detailed information and usage examples.

        Args:
            env: The environment object conforming to the OpenAI Gym interface. It should provide a step function
                and a binary flag indicating collision events within the environment's information dictionary.
            n_h: Number of lookahead steps (predictors) to be used for sample generation and target calculation.
            p_c: Sampling probability for collision samples.
            p_nc: Sampling probability for non-collision samples.
            lambda_val: Lambda value for calculating TD-lambda targets.
            n_stacking: Number of observations to be stacked for creating state representations.
            td_max: Maximum return horizon for TD-lambda calculations (defaults to n_h if not specified).
            controller: A controller object that provides a 'get_action(state)' function. Set to None if no controller is used.
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

        self.lambda_matrix = self._get_lambda_matrix()
        self.eval_samples = None
        self.eval_hist = []

    # =============================================================================================
    # --- Initialize ------------------------------------------------------------------------------
    # =============================================================================================

    def _get_lambda_matrix(self):
        """
            Calculates the lambda matrix used for TD(lambda) target calculation.

            This function generates the lambda matrix, which plays a crucial role in calculating TD(lambda) targets.
            Each row of the matrix corresponds to a specific predictor (lookahead step)
            and contains the corresponding lambda values for calculating the discounted n-step returns.

            The first row has only one non-zero entry, representing the full weight on the immediate reward (collision or non-collision.
            Subsequent rows have increasing numbers of non-zero entries (seconds row has two non-zero entries and so on).
            Each row sums up to 1, ensuring the correct weighting of rewards within the TD(lambda) target calculation.

            Returns:
                NumPy array representing the lambda matrix with dimensions (n_h x td_max), where:
                    - n_h: Number of lookahead steps (predictors).
                    - td_max: Maximum return horizon for TD-lambda calculations.
        """

        # Generate lambda vector
        lambda_vec = (1 - self.lambda_val) * np.array([self.lambda_val ** (power + 1) for power in range(self.td_max)])

        # Replicate for all rows
        lambda_matrix = np.tile(lambda_vec, (self.n_h, 1))

        # Set values that are unreachable for the heads to zero.
        for idx_r in range(self.n_h):
            for idx_c in range(self.td_max):
                if idx_r < idx_c:
                    lambda_matrix[idx_r, idx_c] = 0.0

        # Add remainder to the last element (Sum must be 1)
        for idx_r in range(self.n_h):
            idx_c = min(idx_r, self.td_max - 1)
            lambda_matrix[idx_r, idx_c] += (1 - np.sum(lambda_matrix[idx_r, :]))

        return lambda_matrix

    # =============================================================================================
    # --- Sample Generation Functionality ---------------------------------------------------------
    # =============================================================================================

    def run_episode(self):
        """
            Runs a single episode in the environment and collects observations, actions, rewards, and collision information.

            This function interacts with the environment to simulate an episode. It starts by resetting the environment
            and then iterates through steps until the episode terminates. During each step, it:

            - Retrieves the current state from the environment.
            - If a controller is provided, it queries the controller for an action based on the current state.
            - Stores the current state, action, and reward in the episode data.
            - Executes the action in the environment and receives the next state, reward, and termination information.
            - Repeats these steps until the episode ends due to termination or truncation.

            Finally, the function checks for a collision event based on the environment's information and updates the episode data accordingly.

            Returns:
                A dictionary containing the collected episode data:
                    - observations: List of observed states throughout the episode.
                    - actions: List of actions taken in each step (None for the terminal state).
                    - rewards: List of rewards received in each step.
                    - collision: Boolean indicating whether a collision occurred during the episode.
            """

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
        """
            Processes an episode data dictionary to generate stacked state representations and corresponding samples.

            This function iterates through the observations collected during an episode and creates stacked state representations.
            Each stacked state consists of the last `n_stacking` observations from the episode. It then generates samples
            containing the stacked state, reward received at that state, and the action taken (if applicable).

            Args:
                episode: A dictionary containing episode data generated by the `run_episode` function.

            Returns:
                A dictionary containing processed episode data:
                    - samples: List of samples, each containing:
                        - state: NumPy array representing the stacked state.
                        - reward: Reward received at the corresponding state.
                        - action: Action taken at the corresponding state (None for the terminal state).
                    - collision: Boolean indicating whether a collision occurred during the episode.
        """

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
        """
            Generates TD (Temporal Difference) samples from an episode data dictionary.

            This function processes an episode containing stacked state representations, rewards, and actions to create
            TD samples suitable for TD-learning algorithms. It calculates the number of steps to collision for each
            sample and creates TD samples containing:

            - A sequence of states (up to the maximum lookahead horizon or collision event).
            - Number of steps to the collision event (if applicable, -1 otherwise).
            - A sequence of rewards received in those states.

            Args:
                episode: A dictionary containing processed episode data generated by the `stacking` function.

            Returns:
                A dictionary containing TD samples:
                    - td_samples: List of TD samples, each containing:
                        - states: List of states within the lookahead horizon (up to `self.td_max`).
                        - steps2collision: Number of steps to the next collision event (-1 if not applicable).
                        - rewards: List of rewards received in the corresponding states.
                    - collision: Boolean indicating whether a collision occurred during the episode.
        """

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
        """
            Performs stratified sampling of TD samples from an episode for policy learning.

            This function separates TD samples from a collision episode into collision and non-collision samples. It then
            performs stratified sampling based on the specified probabilities:

            - `self.p_nc`: Probability of sampling non-collision samples.
            - `self.p_c`: Probability of sampling collision samples.

            The function returns a combined list of sampled non-collision and collision samples.

            Args:
                episode: A dictionary containing processed episode data and TD samples generated by the `create_td_samples` function.

            Returns:
                List of sampled TD samples for policy learning.
        """

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
        """
            Generates a specified number of TD samples for policy learning.

            This function iteratively generates episodes by running the environment, processing them into stacked states
            and TD samples, and performing stratified sampling based on collision events. It continues generating episodes
            until the desired number of samples (`n`) is obtained.

            Args:
                n: Number of TD samples to generate.
                return_all: Boolean flag indicating whether to return all generated episode data (default: False).

            Returns:
                If `return_all` is False:
                    A list of `n` randomly sampled TD samples for policy learning.
                If `return_all` is True:
                    A tuple containing:
                        - A list of `n` randomly sampled TD samples for policy learning.
                        - A list of all generated episode data dictionaries.
        """

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

    # =============================================================================================
    # --- Evaluation Functionality ----------------------------------------------------------------
    # =============================================================================================

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

    # =============================================================================================
    # --- Target Generation -----------------------------------------------------------------------
    # =============================================================================================

    def generate_training_data(self, samples: List[Sample], func_inference: Callable):
        """
        Generate training data for collision probability distribution estimation.

        This function calculates the targets for the collision probability distributions based on provided samples
        and an inference function.

        Args:
            samples (List[Sample]): A list of Sample objects containing current states, target states and the information of a collision event.
            func_inference (Callable): A function that returns the current collision probability distribution estimate given the states
                                       with dimension {number of states, *state dimension}.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing the inputs (states) and targets (TD-return) for training the CollisionPro model.
                The last value for each target is the weight for the loss function.
                input dimension :: {number of samples, *state dimension}
                target dimension :: {number of samples, n_h + 1}

        Notes:
            - The inputs represent the states of the environment.
            - The targets represent the fixed-finite TD-return.
            - TD-targets are calculated based on the provided samples and the inference function.

        Detailed Explanation:
        1. Inputs are generated
            - Inputs are the current states of samples and are extracted from the samples
        2. Evaluate target states
            - Evaluate all target/future states that are later used for bootstrapping.
            - First store the target states of all samples in one array and evaluate them with one pass via the provided inference function.
            - In order to redistribute the target states to each sample respectively, a global indexing is introduced.
            - Each sample stores the index of its targets, respectively.
        3. Generate targets (see paper for more information)
            - Let us consider the general idea of the target generation (in case where n_h=td_max):
            - The targets are calculated, where each estimator/head (p_{t->t+i}) is bootstrapping from all (p_{t->t+j}), where j<i.
            - This means that the first head is bootstrapping from one value, the second from two values, the third from three and so on.
            - Next, let us write the update function as follows: P_target{n_h, 1} = SUM( LambdaMatrix{n_h, n_h} ⊙ P_triangular{n_h, n_h}, axis=1)
                - ⊙ is the Hadamard product
                - Lambda is a lower triangular matrix
                - P_triangular is a lower triangular matrix
            - For n_h=3 and td_max=n_h the target generation equation looks like the following:
                | p_{t -> t+1} |      ( | l_{1, 1}        0        0 |   | r_{t+1}            0            0 |          )
                | p_{t -> t+2} | = SUM( | l_{2, 1} l_{2, 2}        0 | ⊙ | p_{t+1->t+2} r_{t+2}            0 | , axis=1 )
                | p_{t -> t+3} |      ( | l_{3, 1} l_{2, 3} l_{3, 3} |   | p_{t+1->t+3} p_{t+2->t+3} r_{t+3} |          )

        Raises:
            ValueError: If the length of samples is less than 1.
        """

        state_dim = samples[0].cur_state.shape
        n_samples = len(samples)

        # =================================================
        # --- 1. Generate inputs --------------------------
        # =================================================

        inputs = np.zeros((n_samples, *state_dim)).squeeze()

        for idx, sample in enumerate(samples):
            inputs[idx] = sample.cur_state

        # =================================================
        # --- 2. Evaluate target states -------------------
        # =================================================

        all_target_states = []
        idx_global = 0

        for sample in samples:
            all_target_states.extend(sample.target_states)

            # Handle indexing
            n_target_states = len(sample.target_states)
            sample.indices = (list(range(idx_global, idx_global + n_target_states)))
            idx_global += n_target_states

        # Calculate for all target states the predicted probability distributions
        all_target_states = np.row_stack(all_target_states).squeeze()
        all_target_probs = func_inference(all_target_states)

        # =================================================
        # --- 3. Generate targets -------------------------
        # =================================================

        # Allocate the targets for the approximator. +1 as the last entry specifies the weighing for the loss.
        targets = np.zeros((n_samples, self.n_h + 1))


        for idx, sample in enumerate(samples):
            # By default, set all values to -1
            p_triangular_matrix = np.tril(-np.ones((self.n_h, self.td_max), dtype=float))

            # Create probability values for each column.
            if sample.is_collision_predecessor():
                for i in range(len(sample.target_states) - 1):
                    p_target_i = all_target_probs[sample.indices[i]]
                    p_triangular_matrix[:, i] = np.concatenate([np.zeros(i + 1), p_target_i[:-(i + 1)]])

            else:
                for i in range(self.td_max):
                    p_target_i = all_target_probs[sample.indices[i]]
                    p_triangular_matrix[:, i] = np.concatenate([np.zeros(i + 1), p_target_i[:-(i + 1)]])

            # Calculate TD-targets
            targets[idx, :-1] = np.sum(self.lambda_matrix * p_triangular_matrix, axis=1)

            # Add weighing for loss function
            targets[idx, -1] = self.p_nc / self.p_c if sample.is_collision_predecessor() else 1.0

        return inputs, targets









