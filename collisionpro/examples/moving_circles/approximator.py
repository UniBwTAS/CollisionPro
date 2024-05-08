import tensorflow as tf
import numpy as np
from collisionpro.core.loss import get_loss_function


class Approximator:
    def __init__(self,
                 n_h,
                 state_dim,
                 epochs=16,
                 batch_size=16,
                 loss_interval=0.5,
                 loss_cumulative=0.5,
                 lr_start=5e-4,
                 lr_decay=0.9,
                 beta_1=0.9,
                 beta_2=0.95):

        self.n_h = n_h
        self.state_dim = state_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_interval = loss_interval
        self.loss_cumulative = loss_cumulative
        self.lr = lr_start
        self.lr_decay = lr_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.optimizer = None
        self.model = self.generate_model()

    def generate_model(self):

        nn_heads = np.array([16] * 16, dtype=int)
        nn_backbones = np.array([32] * 16, dtype=int)
        backbone_res_ctr = 1
        backbone_res_block = 4
        head_res_block = 4

        # =================================================
        # --- Backbone ------------------------------------
        # =================================================

        input_layer = tf.keras.layers.Input(self.state_dim)
        shared_hidden = tf.keras.layers.Dense(nn_backbones[0], activation='elu')(input_layer)
        residual = shared_hidden

        for nn in nn_backbones[1:-1]:

            shared_hidden = tf.keras.layers.Dense(nn, activation='elu')(shared_hidden)
            backbone_res_ctr += 1

            if backbone_res_ctr % backbone_res_block == 0:
                residual = tf.keras.layers.Add()([residual, shared_hidden])
                shared_hidden = residual


        shared_hidden = tf.keras.layers.Dense(nn_backbones[-1], activation='elu')(shared_hidden)

        # =================================================
        # --- Heads ---------------------------------------
        # =================================================

        # List for the outputs of all n-heads
        outputs = []
        branch_output = None

        for idx in range(self.n_h):
            if idx == 0:
                concatenated_predecessor = shared_hidden
            else:
                concatenated_predecessor = tf.keras.layers.Concatenate()([branch_output, shared_hidden])

            branch_hidden = tf.keras.layers.Dense(nn_heads[0], activation='elu')(concatenated_predecessor)
            residual = branch_hidden

            head_res_ctr = 1
            for nn in nn_heads[1:]:
                head_res_ctr += 1

                if head_res_ctr % head_res_block == 0:
                    branch_hidden = tf.keras.layers.Dense(nn, activation='elu')(branch_hidden)
                    residual = tf.keras.layers.Add()([residual, branch_hidden])
                    branch_hidden = residual
                else:
                    branch_hidden = tf.keras.layers.Dense(nn, activation='elu')(branch_hidden)


            branch_output = tf.keras.layers.Dense(1, activation="linear")(branch_hidden)
            outputs.append(branch_output)

        concatenated_outputs = tf.keras.layers.Concatenate()(outputs)

        # =================================================
        # --- Model & Optimizer ---------------------------
        # =================================================

        model = tf.keras.Model(inputs=input_layer, outputs=concatenated_outputs)
        self.optimizer = tf.keras.optimizers.legacy.Nadam(learning_rate=self.lr,
                                                          beta_1=self.beta_1,
                                                          beta_2=self.beta_2)

        loss_function = get_loss_function(const_interval=self.loss_interval,
                                          const_cumulative=self.loss_cumulative,
                                          n_h=self.n_h)
        model.compile(optimizer=self.optimizer, loss=loss_function)

        return model

    def inference(self, inputs):
        return self.model(inputs)

    def fit(self, inputs, targets):
        self.model.fit(inputs,
                      targets,
                      shuffle=True,
                      epochs=self.epochs,
                      validation_split=0.1,
                      batch_size=self.batch_size,
                      )

        # Adjust learning rate
        self.lr = self.lr * self.lr_decay
        self.optimizer.lr.assign(self.lr)


