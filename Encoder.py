import tensorflow as tf
from positionalencoding import positional_encoding
from Attention import MultiHeadAttention
from point_ffn import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, v, k, q, training, mask=None):
        attn_output, _ = self.mha(v, k, q, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        def norm(x):
            nonlocal training
            nonlocal attn_output
            out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
            out2 = self.layernorm2(self.ffn(out1), training=training)
            return self.layernorm2(out1 + out2)

        # (batch_size, input_seq_len, d_model)
        outv = norm(v)
        outk = norm(k)
        outq = norm(q)

        return outv, outk, outq


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate, inp_dim):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.inp_dim = inp_dim
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # TODO: I think the paper actually says positional encoding is added
        # directly to the dimensions of the embedding, not at the end?
        # TODO: 100 is hard coded for bad reasons
        self.pos_encoding = positional_encoding(100, inp_dim)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.inp_dim, tf.float32))

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # TODO: I think this is how it actually works in the paper. I think the
        # OG implementation was flawed because v=k=q in all instances.
        v, k, q = self.ffn(x), self.ffn(x), self.ffn(x)
        for i in range(self.num_layers):
            v, k, q = self.enc_layers[i](v, k, q, training, mask)

        return v  # (batch_size, input_seq_len, d_model
