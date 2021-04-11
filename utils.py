# coding=utf-8
# ====================================
""" Utils.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
from fontTools.ttLib.ttFont import TTFont
import numpy as np
import os
import pandas as pd
import tensorflow as tf

assert tf.__version__.startswith("2")

FLAGS = flags.FLAGS


def net_flags():
    flags.DEFINE_string("dataset_path", "/home/jonnyjack7/fonts/dataset/", " Dataset Folder")
    flags.DEFINE_integer("buffer_size", 100000, "Shuffle buffer size")
    flags.DEFINE_integer(
        "sequence_length", 100, "Maxinum number of words in a sequence"
    )
    flags.DEFINE_integer("epochs", 1000, "Number of Epochs")
    flags.DEFINE_integer("batch_size", 128, "Batch Size")
    flags.DEFINE_integer("per_replica_batch_size", 16, "Batch Size")
    flags.DEFINE_integer("num_layers", 4, "Nnmber of Encoder/Decoder Stack")
    flags.DEFINE_integer(
        "d_model", 128, "Output dimesion of all sublayers including Embedding layer"
    )
    flags.DEFINE_integer("dff", 256, "Dimensionality of inner layer")
    flags.DEFINE_integer("num_heads", 4, "Number of Attention Head")
    flags.DEFINE_boolean("enable_function", False, "Enable Function")
    flags.DEFINE_integer("max_ckpt_keep", 5, "Maximum Number of Checkpoint to keep")
    flags.DEFINE_string("ckpt_path", "model_dist", "Checkpoint Path")
    flags.DEFINE_float("dropout_rate", 0.05, "Dropout Probability")


def flags_dict():
    """Define the flags.

    Returns:
      Command line arguments as Flags.
    """

    kwargs = {
        "dataset_path": FLAGS.dataset_path,
        "enable_function": FLAGS.enable_function,
        "buffer_size": FLAGS.buffer_size,
        "batch_size": FLAGS.batch_size,
        "per_replica_batch_size": FLAGS.per_replica_batch_size,
        "sequence_length": FLAGS.sequence_length,
        "epochs": FLAGS.epochs,
        "num_layers": FLAGS.num_layers,
        "d_model": FLAGS.d_model,
        "dff": FLAGS.dff,
        "num_heads": FLAGS.num_heads,
        "max_ckpt_keep": FLAGS.max_ckpt_keep,
        "ckpt_path": FLAGS.ckpt_path,
        "dropout_rate": FLAGS.dropout_rate,
    }

    return kwargs


def read_data(fontDir):
    """Read *.ttf files and create tf.data.Dataset

    Args:
        fontDir: *.ttf file directory.

    Returns:
        dataset: tf.Data.Dataset of tuples (features, label)

    """
    characters = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrztuvwxyz"  # 52 alphanumeric
    )

    # yields x, y, is_shape_point, is_foreground_glyph, sentinel_glyph_gap
    def generate():
        def glyph_pts(font, g, xOffset=0.0, yOffset=0.0):
            yield np.array([0, 0, 0, 0, 1])
            if g.numberOfContours == -1:
                for comp in g.components:
                    glyph = font.getGlyphSet()[comp.glyphName]._glyph
                    for p in glyph_pts(
                        font, glyph, xOffset=xOffset + comp.x, yOffset=yOffset + comp.y
                    ):
                        yield p
            else:
                foreground = 1
                for i, (x, y) in enumerate(g.coordinates):
                    end_contour = i in g.endPtsOfContours
                    yield np.array([x, y, min(g.flags[i], 1.0), foreground, 0])
                    if end_contour:
                        yield np.array([0, 0, 0, 0, 1])
                    foreground = 0  # only the first contour is "positive" space

        for fontNo, file in enumerate(os.listdir(fontDir)):
            fullpath = os.path.join(fontDir, file)
            if not os.path.isfile(fullpath):
                continue
            print("Font #: %d, %s" % (fontNo, file))
            font = TTFont(fullpath)
            glyphset = font.getGlyphSet()
            for i, c in enumerate(characters):
                if c not in glyphset:
                    continue
                pts = []
                for p in glyph_pts(font, glyphset[c]._glyph):
                    pts.append(p)

                # attention block capped at 100
                if len(pts) > 100:
                    continue
                
                ## normalize (x,y) coords to be 0 < 1, but don't skew letter
                pts = np.array(pts, dtype="float")
                pts[:,0] -= np.min(pts[:,0])
                pts[:,1] -= np.min(pts[:,1])
                pts[:,:2] /= np.max(pts[:,:2])

                # pad the array to 100 points
                features = np.ones([100,5]) * -1e9
                features[:pts.shape[0],:pts.shape[1]] = pts

                lbl = np.zeros(len(characters))
                lbl[i] = 1
                yield features, lbl

    features, labels = [], []
    for f, l in generate():
        features.append(f)
        labels.append(l)
    features = np.array(features)
    labels = np.array(labels)
    return tf.data.Dataset.from_tensor_slices((features,labels))


def load_dataset(
    dataset_path,
    sequence_length,
    batch_size,
    buffer_size,
):
    """Create a tf.data Dataset.

    Args:
        dataset_path: Path to the files to load text from
        sequence_length: Maximun Length of the Sequence
        batch_size: Batch size.
        buffer_size: Buffer size for suffling

    Returns:
        train_dataset: Training dataset
        validate_dataset: validate dataset
    """

    train_dataset = read_data(dataset_path + "train")
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    validate_dataset = read_data(dataset_path + "validate")
    validate_dataset = validate_dataset.batch(batch_size)

    return train_dataset, validate_dataset
