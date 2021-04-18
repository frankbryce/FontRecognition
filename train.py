from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ERROR
import numpy as np
import time
import sys
import pdb
import datetime
from absl import flags
from absl import app

import tensorflow as tf

import utils
from EncoderLabeler import EncoderLabeler

assert tf.__version__.startswith("2")

FLAGS = flags.FLAGS

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=60000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Train(object):
    """Train class

    Args:
        epochs: Number of Epochs
        enable_function: Decorate function with tf.function
        net: EncoderLabeler
        batch_size: Batch size
        train_log_dir: Training log directory
        validate_log_dir: validate Log directory
        max_ckpt_keep: Maximum Number of Checkpoint to keep
        ckpt_path: Checkpoint path
        d_model: Output dimesion of all sublayers including Embedding layer

    """

    def __init__(
        self,
        epochs,
        enable_function,
        net,
        batch_size,
        train_log_dir,
        validate_log_dir,
        max_ckpt_keep,
        ckpt_path,
        d_model,
    ):
        self.epochs = epochs
        self.enable_function = enable_function
        self.net = net
        self.batch_size = batch_size
        self.learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE,
        )
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validate_loss = tf.keras.metrics.Mean(name="validate_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.validate_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="validate_accuracy"
        )

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.validate_summary_writer = tf.summary.create_file_writer(validate_log_dir)
        self.ckpt = tf.train.Checkpoint(
            net=self.net, optimizer=self.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, ckpt_path, max_to_keep=max_ckpt_keep
        )

    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        return tf.reduce_sum(loss_) * 1.0 / self.batch_size

    @tf.function
    def predict(self, features):
        """Greedy Inference

        Args:
            input points: encoded ttf points
        Return:
            result: predicted result of the input sentence
        """
        predictions = self.net.call(features, False)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int64)
        return predicted_id

    def train_step(self, inputs):
        """One Training Step
        Args:
            inputs: Tuple of features, labels tensors
        """
        features, lbls = inputs
        lbls = np.argmax(lbls, axis=1)
        with tf.GradientTape() as tape:
            predictions = self.net(features, training=True)
            loss = self.loss_function(lbls, predictions)
            gradients = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.net.trainable_variables)
            )

        self.train_loss(loss)
        self.train_accuracy(lbls, predictions)

    def validate_step(self, inputs):
        """One validate Step
        Args:
            inputs: Tuple of features, labels tensors
        """
        features, lbls = inputs
        lbls = np.argmax(lbls, axis=1)
        predictions = self.net(features, training=False)

        t_loss = self.loss_function(lbls, predictions)
        self.validate_loss(t_loss)
        self.validate_accuracy(lbls, predictions)

    def load_ckpt(self):
        """if a checkpoint exists, restore the lavalidate checkpoint."""
        status = self.ckpt_manager.restore_or_initialize()
        print(f"ckpt_manager.restore_or_initialize(): {status}")

    def training_loop(self, train_dataset, validate_dataset):
        """Custom training and validateing loop.

        Args:
            train_dataset: Training dataset
            validate_dataset: validateing dataset
        """

        if self.enable_function:
            self.train_step = tf.function(self.train_step)
            self.validate_step = tf.function(self.validate_step)
        template = "Epoch {}  Loss {:.4f} Accuracy {:.4f}, validate Loss {:.4f}, validate Accuracy {:.4f}"

        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.validate_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validate_accuracy.reset_states()

            start = time.time()
            counter = 0

            print(len(train_dataset))
            
            for src, tgt in train_dataset:
                self.train_step((src, tgt))
                counter += 1
                if (counter + 1) % 100 == 0 or FLAGS.testing:
                    for t_src, t_tgt in validate_dataset:
                        self.validate_step((t_src, t_tgt))
                    print(
                        template.format(
                            epoch + 1,
                            self.train_loss.result(),
                            self.train_accuracy.result() * 100,
                            self.validate_loss.result(),
                            self.validate_accuracy.result() * 100,
                        )
                    )

            with self.train_summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", self.train_accuracy.result(), step=epoch)

            with self.validate_summary_writer.as_default():
                tf.summary.scalar("loss", self.validate_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", self.validate_accuracy.result(), step=epoch)

            ckpt_save_path = self.ckpt_manager.save()
            print(
                "Saving checkpoint for epoch {} at {}".format(
                    epoch + 1, ckpt_save_path
                )
            )

            print(
                "Time taken for {} epoch: {} secs\n".format(
                    epoch + 1, (time.time() - start)
                )
            )


def run_main(argv):
    del argv
    kwargs = utils.flags_dict()
    del kwargs["per_replica_batch_size"]
    main(**kwargs)


def main(
    epochs,
    enable_function,
    buffer_size,
    batch_size,
    d_model,
    dff,
    num_heads,
    dataset_path,
    dropout_rate,
    num_layers,
    sequence_length,
    ckpt_path,
    max_ckpt_keep,
):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/gradient_tape/" + current_time + "/train"
    validate_log_dir = "logs/gradient_tape/" + current_time + "/validate"
    num_labels = 52

    train_dataset, validate_dataset = utils.load_dataset(
        dataset_path,
        sequence_length,
        batch_size,
        buffer_size,
    )
    net = EncoderLabeler(
        num_layers,
        d_model,
        num_heads,
        dff,
        num_labels,
        dropout_rate,
    )

    print("create training object")
    train_obj = Train(
        epochs,
        enable_function,
        net,
        batch_size,
        train_log_dir,
        validate_log_dir,
        max_ckpt_keep,
        ckpt_path,
        d_model,
    )

    train_obj.training_loop(train_dataset, validate_dataset)
    train_obj.load_ckpt()
    tf.saved_model.save(train_obj.net, "model")


if __name__ == "__main__":
    utils.net_flags()
    app.run(run_main)
