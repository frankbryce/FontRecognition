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
from tqdm import trange

assert tf.__version__.startswith("2")

FLAGS = flags.FLAGS


def net_flags():
    flags.DEFINE_string("dataset_path", "dataset/", " Dataset Folder")
    flags.DEFINE_integer(
        "sequence_length", 100, "Maxinum number of words in a sequence"
    )
    flags.DEFINE_integer("epochs", 1000, "Number of Epochs")
    flags.DEFINE_integer("batch_size", 128, "Batch Size")
    flags.DEFINE_integer("per_replica_batch_size", 16, "Batch Size")
    flags.DEFINE_integer("num_layers", 1, "Number of Encoder/Decoder Stack")
    flags.DEFINE_integer(
        "d_model", 64, "Output dimesion of all sublayers including Embedding layer"
    )
    flags.DEFINE_integer("dff", 128, "Dimensionality of inner layer")
    flags.DEFINE_integer("num_heads", 8, "Number of Attention Head")
    flags.DEFINE_boolean("enable_function", False, "Enable Function")
    flags.DEFINE_integer("max_ckpt_keep", 5, "Maximum Number of Checkpoint to keep")
    flags.DEFINE_string("ckpt_path", "model_dist", "Checkpoint Path")
    flags.DEFINE_float("dropout_rate", 0.1, "Dropout Probability")
    flags.DEFINE_bool("testing", False,
            ("Used to test locally on laptop, before uploading to github to run "
             "in colab. This loads only one test point, and expedites testing "
             "of the mechanics of the environment rather than focusing on "
             "training accuracy."))


def flags_dict():
    """Define the flags.

    Returns:
      Command line arguments as Flags.
    """

    kwargs = {
        "dataset_path": FLAGS.dataset_path,
        "enable_function": FLAGS.enable_function,
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

def getTiling(
        dimensions,
        tiles_per_dim,
        coords,
        min_value=0.0,
        max_value=1.0):
    try:
      ind = 0
      tile_length = (max_value-min_value) / tiles_per_dim
      total_tiles = tiles_per_dim ** dimensions
      for dim in range(dimensions):
          dim_ind = min(tiles_per_dim-1, int((coords[dim]-min_value) / tile_length))
          ind += dim_ind * (tiles_per_dim**dim)
      tiling = np.zeros(total_tiles)
      tiling[ind] = 1
      return tiling
    except:
      print(dimensions,tiles_per_dim,coords,min_value,max_value)
      raise

def getTilings(
        dimensions,
        tiles_per_dim,
        num_tilings,
        coords,
        min_value=0.0,
        max_value=1.0):
    tiles_per_tiling = tiles_per_dim**dimensions
    tile_length = (max_value-min_value) / tiles_per_dim
    tilings = np.empty([num_tilings,tiles_per_dim ** dimensions])
    coords = np.array(coords)
    for i in range(num_tilings):
        offset = (tile_length * i * 141 / num_tilings) % tile_length
        tiling = getTiling(
                dimensions,
                tiles_per_dim,
                coords + offset,
                min_value,
                max_value+tile_length)
        tilings[i] = tiling
    return list(np.ndarray.flatten(tilings))

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
    def generate(flag_min=0.0, flag_max=1.0):
        def _tiling(x,y,bb):
            x -= bb[0]
            x /= (bb[2]-bb[0])
            y -= bb[1]
            y /= (bb[3]-bb[1])
            return getTilings(2, 10, 10, np.array([x,y]))
        def raw_pts(font, g, xOffset=0.0, yOffset=0.0):
            if g.numberOfContours == -1:
                for comp in g.components:
                    glyph = font.getGlyphSet()[comp.glyphName]._glyph
                    for p in raw_pts(
                        font, glyph, xOffset=xOffset + comp.x, yOffset=yOffset + comp.y
                    ):
                        yield p
            else:
                for i, p in enumerate(g.coordinates):
                    yield p

        def glyph_pts(font, g, bb, xOffset=0.0, yOffset=0.0):
            # TODO: hardcoded 1000
            yield np.array([0]*1000 + [0, 0, flag_max])
            if g.numberOfContours == -1:
                for comp in g.components:
                    glyph = font.getGlyphSet()[comp.glyphName]._glyph
                    for p in glyph_pts(
                        font, glyph, bb, xOffset=xOffset + comp.x, yOffset=yOffset + comp.y
                    ):
                        yield p
            else:
                foreground = flag_max
                for i, (x, y) in enumerate(g.coordinates):
                    end_contour = i in g.endPtsOfContours
                    yield np.array(_tiling(x,y,bb) + [
                        max(min(g.flags[i], 1.0), 0.0)/10.0 + flag_min, foreground, flag_min])
                    if end_contour:
                        # TODO: hardcoded 1000
                        yield np.array([0]*1000 + [0, 0, flag_max])
                    foreground = flag_min  # only the first contour is "positive" space

        ttf_files = os.listdir(fontDir)
        for fontNo in trange(len(ttf_files)):
            file = ttf_files[fontNo]
            fullpath = os.path.join(fontDir, file)
            if not os.path.isfile(fullpath):
                continue
            # print("Font #: %d, %s" % (fontNo, file))
            font = TTFont(fullpath)
            glyphset = font.getGlyphSet()
            for i, c in enumerate(characters):
                if c not in glyphset:
                    continue
                raw = []
                for p in raw_pts(font, glyphset[c]._glyph):
                    raw.append(p)
                raw = np.array(raw, dtype="float")
                bb = (np.min(raw[:,0]),np.min(raw[:,1]),np.max(raw[:,0]),np.max(raw[:,1]))

                pts = []
                for p in glyph_pts(font, glyphset[c]._glyph, bb):
                    pts.append(p)

                # attention block capped at 100
                if len(pts) > 100:
                    continue
                
                # normalize (x,y) coords to be 0 < 1, but don't skew letter
                pts = np.array(pts, dtype="float")

                # pad the array to 100 points
                features = np.ones([100,1003]) * -1e9
                features[:pts.shape[0],:pts.shape[1]] = pts

                lbl = np.zeros(len(characters))
                lbl[i] = 1
                yield features, lbl

            # if we're testing, only return one font to expedite testing
            if FLAGS.testing and fontNo >= 9:
                return

    features, labels = [], []
    for f, l in generate():
        features.append(f)
        labels.append(l)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)


def load_dataset(
    dataset_path,
    sequence_length,
    batch_size,
):
    """Create a tf.data Dataset.

    Args:
        dataset_path: Path to the files to load text from
        sequence_length: Maximun Length of the Sequence
        batch_size: Batch size.

    Returns:
        train_dataset: Training dataset
        validate_dataset: validate dataset
    """

    print("loading train_dataset...")
    train_dataset = read_data(dataset_path + "train")
    print("loading validate_dataset...")
    validate_dataset = read_data(dataset_path + "validate")

    return train_dataset, validate_dataset
