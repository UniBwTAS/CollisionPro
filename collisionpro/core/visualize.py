import matplotlib.pyplot as plt
import numpy as np
import os


def create_collision_characteristics(func_inference,
                                     collision_pro,
                                     dt,
                                     kind="mean",
                                     num=25,
                                     path=None):
    """
    TODO :: Desired path

    :param func_inference:
    :param collision_pro:
    :param kind: Either "mean", "both" or "examples"
    :param num: Number of characteristics
    :param dt: Time step value
    :param path: path to file
    :return:
    """

    characteristics = np.zeros((num, collision_pro.n_h, collision_pro.n_h))

    for idx in range(num):
        episode = collision_pro.run_episode()
        episode = collision_pro.stacking(episode)
        episode = collision_pro.create_td_samples(episode)
        pre_coll_samples = episode["td_samples"][-collision_pro.n_h:]
        pre_coll_states = []
        for sample in pre_coll_samples:
            pre_coll_states.append(sample.cur_state)
        pre_coll_states = np.row_stack(pre_coll_states)

        characteristics[idx, :, :] = -func_inference(pre_coll_states)

    outputs = []

    if kind in ["mean", "both"]:
        mean_char = np.mean(characteristics, axis=0)
        outputs.append({"name": "mean", "characteristic": mean_char})

    if kind in ["examples", "both"]:
        for idx in range(num):
            mean_char = np.mean(characteristics, axis=0)
            outputs.append({"name": f"example_{idx}", "characteristic": characteristics[idx, :, :]})

    x = np.linspace(dt, dt * collision_pro.n_h, collision_pro.n_h)
    y = np.linspace(dt, dt * collision_pro.n_h, collision_pro.n_h)
    X, Y = np.meshgrid(x, y)

    path = "" if path is None else path

    for ele in outputs:
        fig = plt.figure()
        ax_ = fig.add_subplot(111, projection='3d')

        surf = ax_.plot_surface(X, Y, ele["characteristic"], cmap='RdYlGn_r')

        ax_.set_xlabel('Horizon [s]')
        ax_.set_ylabel('Time [s]')
        ax_.set_zlabel('p')

        fig.show()
        fig.savefig(f'{os.path.join(path, ele["name"])}.pdf')
