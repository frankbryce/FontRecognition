import tensorflow as tf
from Encoder import Encoder
from positionalencoding import create_masks


class EncoderLabeler(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        num_labels,
        rate=0.1,
    ):
        super(EncoderLabeler, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=num_labels)
        self.softmax = tf.keras.layers.Softmax()

    # @tf.function(input_signature=[tf.TensorSpec([None,None],tf.int64,name='inp'),
    #                               tf.TensorSpec(None,tf.bool,name='training')])
    def call(self, inp, training=False):
        enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)
        return self.softmax(self.dense(self.flatten(enc_output)))
