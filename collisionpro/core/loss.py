import tensorflow as tf


def get_loss_function(n_h, const_interval=1.0, const_cumulative=1.0):
    """
    This loss function is described in the paper.
    It was designed for high sample efficiency and unbiased probability estimation.
    The loss function computes an MSE loss and a combination of interval and cumulative losses, with adjustable constants for each component.

    Dimensions:
        y_true :: (batch size, *state dimension)
        y_pred :: (batch_size, N_h + 1)
                  - The last element for y_pred is the weighing factor collision and non-collision samples.

    Parameters:
        :param n_h: Number of lookahead steps
        :param const_interval: Constant multiplier for interval loss (being in the range [0; 1]
        :param const_cumulative: Constant multiplier for cumulative loss (distribution must be strictly increasing)
    """

    def loss_function(y_true, y_pred):

        # =================================================
        # --- Weighted MSE --------------------------------
        # =================================================

        loss = tf.reduce_mean(tf.multiply(tf.tile(tf.expand_dims(y_true[:, -1], axis=1),
                                                  tf.constant([1, n_h], tf.int32)),
                                          tf.square(y_true[:, :-1] - y_pred)), axis=0)

        # ================================================
        # Cumulative probability distribution related loss
        # ================================================

        # 1. Should be in interval [0; 1]
        # ===============================

        # Create a mask for values greater than 0 and smaller than -1
        greater_than_zero_mask = tf.greater(y_pred, 0)
        smaller_than_minus_one_mask = tf.less(y_pred, -1)

        # Compute the loss based on the masks
        loss_prob_interval = tf.where(greater_than_zero_mask, tf.abs(-y_pred), 0.0)
        loss_prob_interval = tf.where(smaller_than_minus_one_mask, tf.abs(-1 - y_pred), loss_prob_interval)
        loss_prob_interval = const_interval * tf.reduce_mean(tf.square(loss_prob_interval), axis=0)

        # 2. The cumulative values should strictly increase :: p_{t+i+1} >= p_{t+i}
        # =========================================================================

        prob_diff = y_pred[:, 1:] - y_pred[:, :-1]
        loss_prob_cumulative = tf.where(prob_diff > 0, tf.zeros_like(prob_diff), prob_diff)
        zero_column = tf.zeros((tf.shape(y_pred)[0], 1), dtype=tf.float32)
        loss_prob_cumulative_forward = tf.concat([loss_prob_cumulative, zero_column], axis=1)
        loss_prob_cumulative_backward = tf.concat([zero_column, loss_prob_cumulative], axis=1)
        loss_prob_cumulative = tf.square(loss_prob_cumulative_forward - loss_prob_cumulative_backward)
        loss_prob_cumulative = tf.reduce_mean(loss_prob_cumulative, axis=0)
        loss_prob_cumulative = const_cumulative * loss_prob_cumulative

        # ================================================
        # --- Calc. total loss ---------------------------
        # ================================================

        total_loss = loss + loss_prob_interval + loss_prob_cumulative

        return total_loss

    return loss_function
