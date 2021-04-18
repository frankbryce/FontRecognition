import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff/2, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model, activation="relu"),  # (batch_size, seq_len, d_model)
        ]
    )
