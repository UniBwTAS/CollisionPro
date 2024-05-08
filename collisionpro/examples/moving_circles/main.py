from collisionpro.examples.moving_circles.env import MovingCircles
from collisionpro.examples.moving_circles.approximator import Approximator
from collisionpro.examples.moving_circles.controller import Controller
from collisionpro.core.collisionpro import CollisionPro
from collisionpro.core.visualize import create_collision_characteristics

import numpy as np
import matplotlib.pyplot as plt

def run(n_h=20,
        td_max=10,
        p_c=1.0,
        p_nc=0.2,
        n_training_cycles=15,
        n_samp_total=2500,
        n_stacking=1,
        lr_start=2e-4,
        lr_decay=0.999,
        lambda_val=0.7,
        batch_size=32,
        epochs=16,
        num_collision_characteristics=5,
        save_figures=False,
        path=None):

    # =========================================================
    # --- Initialization --------------------------------------
    # =========================================================


    env_moving_circles = MovingCircles(max_obstacles=3)
    env_moving_circles.reset()

    controller = Controller(env_moving_circles)

    approximator = Approximator(n_h=n_h,
                                state_dim=env_moving_circles.state.shape,
                                lr_start=lr_start,
                                lr_decay=lr_decay,
                                batch_size=batch_size,
                                epochs=epochs)

    collision_pro = CollisionPro(env=env_moving_circles,
                                 p_c=p_c,
                                 p_nc=p_nc,
                                 n_h=n_h,
                                 lambda_val=lambda_val,
                                 td_max=td_max,
                                 n_stacking=n_stacking,
                                 controller=controller)

    evaluation_samples = collision_pro.generate_evaluation_samples(10000, p_s=0.1)
    collision_pro.set_evaluation_samples(evaluation_samples)
    err_acc = []
    err_pes = []

    # =========================================================
    # --- Training --------------------------------------------
    # =========================================================

    for idx in range(n_training_cycles):
        samples = collision_pro.generate_samples(n_samp_total)
        inputs, targets = collision_pro.generate_training_data(samples, approximator.inference)
        approximator.fit(inputs, targets)

        cur_err_ass, cur_err_pes = collision_pro.evaluate(approximator.inference)

        err_acc.append(np.mean(cur_err_ass))
        err_pes.append(np.mean(cur_err_pes))

        print(f"Cycle [{idx + 1}/{n_training_cycles}]")

    # =========================================================
    # --- Make Collision Characteristics ----------------------
    # =========================================================


    create_collision_characteristics(func_inference=approximator.inference,
                                     collision_pro=collision_pro,
                                     kind="both",
                                     num=num_collision_characteristics,
                                     dt=env_moving_circles.dt,
                                     save_figures=save_figures,
                                     path=path)

    # =========================================================
    # --- Plot Learning Performance ---------------------------
    # =========================================================

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Accuracy")
    ax.set_xlabel("Learning steps")
    ax.set_ylabel("Error")
    ax.plot(err_acc, 'b-o')
    ax.grid()
    fig.show()

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Pessimism")
    ax.set_xlabel("Learning steps")
    ax.set_ylabel("Error")
    ax.plot(err_pes, 'b-o')
    ax.grid()
    fig.show()

    plt.show()

    # =========================================================
    # --- Animate ---------------------------------------------
    # =========================================================

    # env_moving_circles.reset()
    # env_moving_circles.rendering(controller, delta_time=0.025)


if __name__ == "__main__":
    run()

