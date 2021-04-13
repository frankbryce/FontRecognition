import tensorflow as tf
from positionalencoding import positional_encoding
from Attention import MultiHeadAttention
from point_ffn import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads,
                d_model,
                dropout=rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        attn_output = self.mha(x, x)  # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output, training=training)  # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(self.ffn(out1), training=training)
        return self.layernorm3(out1 + out2, training=training)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # TODO: 100 is hard coded for bad reasons
        self.pos_encoding = positional_encoding(100, d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        x = self.ffn(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x  # (batch_size, input_seq_len, d_model
