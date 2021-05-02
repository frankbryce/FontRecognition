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
        self.ckpt = tf.train.Checkpoint(
            net=self.net,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, ckpt_path, max_to_keep=max_ckpt_keep
        )

    def load_ckpt(self):
        """if a checkpoint exists, restore the lavalidate checkpoint."""
        status = self.ckpt_manager.restore_or_initialize()
        print(f"ckpt_manager.restore_or_initialize(): {status}")

    def training_loop(self, train_data, train_lbls, validate_data,
            validate_lbls):
        """Custom training and validateing loop.

        Args:
            train_data: Training raw inputs
            train_lbls: 1 hot encoded labels
            validate_data: validation raw inputs
            validate_lbls: 1 hot encoded labels
        """
        self.net.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        self.net.fit(
                train_data,
                train_lbls,
                epochs=self.epochs,
                batch_size=self.batch_size)
        train_loss, train_acc = self.net.evaluate(train_data, train_lbls)
        validate_loss, validate_acc = self.net.evaluate(validate_data, validate_lbls)


def run_main(argv):
    del argv
    kwargs = utils.flags_dict()
    del kwargs["per_replica_batch_size"]
    main(**kwargs)


def main(
    epochs,
    enable_function,
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

    (train_data, train_lbls), (validate_data, validate_lbls) = utils.load_dataset(
        dataset_path,
        sequence_length,
        batch_size,
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

    train_obj.training_loop(
            train_data,
            train_lbls,
            validate_data,
            validate_lbls)
    train_obj.load_ckpt()
    tf.saved_model.save(train_obj.net, "model")


if __name__ == "__main__":
    utils.net_flags()
    app.run(run_main)
