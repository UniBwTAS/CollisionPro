import matplotlib.pyplot as plt
import numpy as np
import os


def create_collision_characteristics(func_inference,
                                     collision_pro,
                                     dt,
                                     kind="mean",
                                     num=25,
                                     save_figures=False,
                                     path=None):
    """
    Creates collision characteristics based on provided collision probabilities.
    Collision characteristics are the collision probability distributions within the last n_h steps.

    Parameters:
    ----------
    func_inference : function
        A function that takes in an array of states (n_h, *state_dimension) and returns inferred collision probabilities.
    collision_pro : CollisionPro
        An instance of CollisionPro class providing methods for generating episodes.
    dt : float
        Time interval between consecutive steps in seconds.
    kind : str, optional
        The type of characteristics to generate. Possible values are 'mean', 'examples', or 'both'. Default is 'mean'.
    num : int, optional
        The number of episodes to generate for calculating characteristics. Default is 25.
    save_figures : bool, optional
        If True, generated figures will be saved to disk. Default is False.
    path : str, optional
        The absolute path to save the figures. If None, figures will be saved in the current directory. Default is None.

    Returns:
    -------
    None

    Notes:
    ------
    - The function generates collision characteristics by running multiple episodes and extracting the last n_h steps
      of the collision probability distribution estimation.
    - Characteristics are calculated using the provided function for inference.
    - Figures can be generated for mean characteristics, individual examples, or both, depending on the 'kind' parameter.
    - If collision happens before n_h steps, function terminates.

    Example:
    --------
    >>> from collisionpro.core.collisionpro import CollisionPro
    >>> from collisionpro.examples.moving_circles.approximator import Approximator
    >>> create_collision_characteristics(Approximator.inference, collision_pro, dt=0.1, kind='both', num=50, save_figures=True, path='/path/to/save')
    """

    # =========================================================================
    # --- Create Collision Characteristics ------------------------------------
    # =========================================================================

    # Allocate
    characteristics = np.zeros((num, collision_pro.n_h, collision_pro.n_h))

    # Generate episodes and extract the last n_h steps of the collision probability distribution estimation.
    for idx in range(num):

        # Generate sample
        episode = collision_pro.run_episode()
        episode = collision_pro.create_td_samples(episode)

        # Check if episode is long enough
        if len(episode["td_samples"]) < collision_pro.n_h:
            print("Collision characteristic generation terminates as length of episode was shorter than n_h!")
            return

        # Get pre-collision samples
        pre_coll_samples = episode["td_samples"][-collision_pro.n_h:]

        # Get numpy array of states -> Dimension :: (n_h, n_h)
        pre_coll_states = []
        for sample in pre_coll_samples:
            pre_coll_states.append(sample.cur_state)
        pre_coll_states = np.row_stack(pre_coll_states)

        # Save characteristic and reverse sign to get meaningful probability values
        characteristics[idx, :, :] = -func_inference(pre_coll_states)

    # =========================================================================
    # --- Generate Figures ----------------------------------------------------
    # =========================================================================

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

    path = os.getcwd() if path is None else path
    for ele in outputs:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, ele["characteristic"], cmap='RdYlGn_r')

        ax.set_xlabel('Horizon [s]')
        ax.set_ylabel('Time [s]')
        ax.set_zlabel('p')
        ax.set_title(ele["name"])

        if save_figures:
            fig.show()
            fig.savefig(f'{os.path.join(path, ele["name"])}.pdf')
        else:
            fig.show()

    if not save_figures:
        plt.show()

