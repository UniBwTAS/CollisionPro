from collisionpro.examples.random_walk.env import RandomWalk
from collisionpro.examples.random_walk.approximator import Approximator
from collisionpro.core.collisionpro import CollisionPro

import matplotlib.pyplot as plt
import numpy as np


def run(n_h=5,
        td_max=5,
        p_c=1.0,
        p_nc=1.0,
        lambda_val=0.25,
        n_training_cycles=50,
        n_samp_total=2000,
        alpha=0.1,
        alpha_decay=0.9,
        n_states=7):


    # =========================================================
    # --- Initialization --------------------------------------
    # =========================================================


    env_random_walk = RandomWalk(n_states=n_states)

    approximator = Approximator(n_states=n_states,
                                n_h=n_h,
                                alpha=alpha,
                                alpha_decrease=alpha_decay)

    collision_pro_ = CollisionPro(p_c=p_c,
                                  p_nc=p_nc,
                                  n_h=n_h,
                                  env=env_random_walk,
                                  lambda_val=lambda_val,
                                  td_max=td_max)


    # =========================================================
    # --- Training --------------------------------------------
    # =========================================================


    for idx in range(n_training_cycles):
        samples = collision_pro_.generate_samples(n_samp_total)
        inputs, targets = collision_pro_.generate_training_data(samples, approximator.inference)
        approximator.fit(inputs, targets)

        print(f"Cycle [{idx + 1}/{n_training_cycles}]")


    # =========================================================
    # --- Visualization ---------------------------------------
    # =========================================================


    # Probabilities are the negative values of the predictions
    probabilities = -approximator.table.T
    probabilities[probabilities < 1e-4] = 0.0

    plt.imshow(probabilities, cmap="RdYlGn_r", interpolation='none')

    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            plt.text(j, i, f'{probabilities[i, j]:.3f}', ha='center', va='center', color='black')

    y_ticks_labels = [f"p (t→t+{idx+1})" for idx in range(n_h)]
    plt.yticks(np.arange(len(y_ticks_labels)), y_ticks_labels)
    plt.xlabel('State')
    plt.ylabel('Cumulative Collision Probability')
    plt.title('Collision Probability Distribution for Random Walk')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()