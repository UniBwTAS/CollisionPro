from collisionpro.examples.moving_circles.env import MovingCircles
from collisionpro.examples.moving_circles.approximator import Approximator
from collisionpro.examples.moving_circles.controller import Controller
from collisionpro.core.collisionpro import CollisionPro
from collisionpro.core.visualize import create_collision_characteristics

import numpy as np
import matplotlib.pyplot as plt


def run(n_h=40,
        td_max=15,
        p_c=0.1,
        p_nc=0.05,
        n_training_cycles=15,
        n_samp_total=1000,
        lr_start=5e-4,
        lr_decay=0.7,
        loss_cumulative=1.0,
        loss_interval=1.0,
        lambda_val=0.7,
        batch_size=64,
        epochs=16,
        num_collision_characteristics=20,
        save_figures=False,
        path=None):

    print("The learning process may require some time, the duration of which depends on your hardware specifications.")

    # =========================================================
    # --- Initialization --------------------------------------
    # =========================================================


    env_moving_circles = MovingCircles(dt=0.2)
    env_moving_circles.reset()

    controller = Controller(env_moving_circles)

    approximator = Approximator(n_h=n_h,
                                state_dim=env_moving_circles.state.shape,
                                lr_start=lr_start,
                                lr_decay=lr_decay,
                                batch_size=batch_size,
                                epochs=epochs,
                                loss_interval=loss_interval,
                                loss_cumulative=loss_cumulative)

    collision_pro = CollisionPro(env=env_moving_circles,
                                 p_c=p_c,
                                 p_nc=p_nc,
                                 n_h=n_h,
                                 lambda_val=lambda_val,
                                 td_max=td_max,
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

    # Uncomment to start rendering for Moving Circles
    env_moving_circles.reset()
    env_moving_circles.rendering(controller, delta_time=0.025, inference=approximator.inference)


if __name__ == "__main__":
    run()

