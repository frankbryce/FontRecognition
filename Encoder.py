import tensorflow as tf
from positionalencoding import positional_encoding
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
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.input_emb = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff*4, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff*2, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff) 
            tf.keras.layers.Dense(d_model, activation="relu"),  # (batch_size, seq_len, d_model)
        ])

        # TODO: 100 is hard coded for bad reasons
        self.pos_encoding = tf.keras.layers.Embedding(input_dim=100, output_dim=d_model)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_encoding(positions)

        x = self.input_emb(x)
        x = x + positions
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x  # (batch_size, input_seq_len, d_model
