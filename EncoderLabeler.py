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

        # TODO: encode inp_dim
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate, 5)
        self.num_labels = num_labels

    # @tf.function(input_signature=[tf.TensorSpec([None,None],tf.int64,name='inp'),
    #                              tf.TensorSpec([None,None],tf.int64,name='out'),
    #                              tf.TensorSpec(None,tf.bool,name='flag')])
    def call(self, inp, training=False):
        enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

        # I think this is what makes sense, though potentially d_model needs to
        # get much smaller..
        o1 = tf.keras.layers.Flatten()(enc_output)
        o2 = tf.keras.layers.Dense(units=self.num_labels, activation='relu')(o1)
        final_output = tf.keras.layers.Softmax()(o2)
        # (batch_size, tar_seq_len, num_labels)

        return final_output
