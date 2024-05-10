import tensorflow as tf
import numpy as np


def generate_model(n_h,
                   state_dim,
                   n_neurons_backbone=256,
                   n_neurons_heads=128,
                   n_layers_backbone=16,
                   n_layers_heads=8,
                   n_skip_block_backbone=4,
                   n_skip_block_heads=4):

    """
    This function provides a tensorflow model for the collision probability distribution approximator.
    The output is defined by the number of heads (number of lookahead steps n_h).


    Args:
        n_h: Number of lookahead steps
        state_dim: State dimension
        n_neurons_backbone: Number of neurons for each backbone layer
        n_neurons_heads: Number of neurons for each head layer
        n_layers_backbone: Number of backbone layers
        n_layers_heads: Number of head layers
        n_skip_block_backbone: Number of layers that are connected by a residual layer for the backbone
        n_skip_block_heads: Number of layers that are connected by a residual layer for the heads

    Return:
         A tensorflow model
    """

    nn_heads = np.array([n_neurons_heads] * n_layers_heads, dtype=int)
    nn_backbones = np.array([n_neurons_backbone] * n_layers_backbone, dtype=int)
    head_res_block = 4

    # =================================================
    # --- Backbone ------------------------------------
    # =================================================

    input_layer = tf.keras.layers.Input(state_dim)
    shared_hidden = tf.keras.layers.Dense(nn_backbones[0], activation='elu')(input_layer)
    residual = shared_hidden
    backbone_res_ctr = 1

    for nn in nn_backbones[1:-1]:

        shared_hidden = tf.keras.layers.Dense(nn, activation='elu')(shared_hidden)
        backbone_res_ctr += 1

        if backbone_res_ctr % n_skip_block_backbone == 0:
            residual = tf.keras.layers.Add()([residual, shared_hidden])
            shared_hidden = residual


    shared_hidden = tf.keras.layers.Dense(nn_backbones[-1], activation='elu')(shared_hidden)

    # =================================================
    # --- Heads ---------------------------------------
    # =================================================

    # List for the outputs of all n-heads
    outputs = []
    branch_output = None

    for idx in range(n_h):
        if idx == 0:
            concatenated_predecessor = shared_hidden
        else:
            concatenated_predecessor = tf.keras.layers.Concatenate()([branch_output, shared_hidden])

        branch_hidden = tf.keras.layers.Dense(nn_heads[0], activation='elu')(concatenated_predecessor)
        residual = branch_hidden

        head_res_ctr = 1
        for nn in nn_heads[1:]:
            head_res_ctr += 1

            if head_res_ctr % n_skip_block_heads == 0:
                branch_hidden = tf.keras.layers.Dense(nn, activation='elu')(branch_hidden)
                residual = tf.keras.layers.Add()([residual, branch_hidden])
                branch_hidden = residual
            else:
                branch_hidden = tf.keras.layers.Dense(nn, activation='elu')(branch_hidden)


        branch_output = tf.keras.layers.Dense(1, activation="linear")(branch_hidden)
        outputs.append(branch_output)

    concatenated_outputs = tf.keras.layers.Concatenate()(outputs)

    # =================================================
    # --- Model ---------------------------------------
    # =================================================

    model = tf.keras.Model(inputs=input_layer, outputs=concatenated_outputs)

    return model
