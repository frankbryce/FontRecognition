import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff, dropout):
    regularizer=tf.keras.regularizers.l1_l2(l1=1e-6, l2=1e-5)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dff,
                activation='relu',
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                activity_regularizer=regularizer),
            tf.keras.layers.Dense(
                d_model,
                activation='relu',
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                activity_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout),
        ]
    )
