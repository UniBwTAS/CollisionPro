import tensorflow as tf
import numpy as np
from collisionpro.core.loss import get_loss_function
from collisionpro.core.nn_model_generation import generate_model


class Approximator:
    def __init__(self,
                 n_h,
                 state_dim,
                 epochs=16,
                 batch_size=16,
                 loss_interval=0.5,
                 loss_cumulative=0.5,
                 lr_start=2e-4,
                 lr_decay=0.875,
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

        model = generate_model(self.n_h, self.state_dim)

        loss_function = get_loss_function(const_interval=self.loss_interval,
                                          const_cumulative=self.loss_cumulative,
                                          n_h=self.n_h)

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr,
                                                         beta_1=self.beta_1,
                                                         beta_2=self.beta_2)

        model.compile(optimizer=self.optimizer, loss=loss_function)

        return model

    def inference(self, inputs):
        verbose = 2 if inputs.shape[0] > 2000 else 1
        return self.model.predict(inputs, batch_size=256, verbose=verbose)

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


