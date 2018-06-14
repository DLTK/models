# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from dltk.core.metrics import dice
from dltk.core.losses import sparse_balanced_crossentropy
from dltk.networks.segmentation.unet import residual_unet_3d
from dltk.networks.segmentation.unet import asymmetric_residual_unet_3d
from dltk.networks.segmentation.fcn import residual_fcn_3d
from dltk.core.activations import leaky_relu
from dltk.io.abstract_reader import Reader
from reader import read_fn
import json

# PARAMS
EVAL_EVERY_N_STEPS = 1000
EVAL_STEPS = 1

NUM_CLASSES = 14
NUM_CHANNELS = 1

BATCH_SIZE = 4
SHUFFLE_CACHE_SIZE = 128

MAX_STEPS = 100000


# MODEL
def model_fn(features, labels, mode, params):
    """Summary

    Args:
        features (TYPE): Description
        labels (TYPE): Description
        mode (TYPE): Description
        params (TYPE): Description

    Returns:
        TYPE: Description
    """
    # 1. create a model and its outputs

    filters = params["filters"]
    strides = params["strides"]
    num_residual_units = params["num_residual_units"]
    loss_type = params["loss"]
    net = params["net"]

    def lrelu(x):
        return leaky_relu(x, 0.1)

    if net == 'fcn':
        net_output_ops = residual_fcn_3d(
            features['x'], NUM_CLASSES,
            num_res_units=num_residual_units,
            filters=filters,
            strides=strides,
            activation=lrelu,
            mode=mode)
    elif net == 'unet':
        net_output_ops = residual_unet_3d(
            features['x'], NUM_CLASSES,
            num_res_units=num_residual_units,
            filters=filters,
            strides=strides,
            activation=lrelu,
            mode=mode)
    elif net == 'asym_unet':
        net_output_ops = asymmetric_residual_unet_3d(
            features['x'],
            NUM_CLASSES,
            num_res_units=num_residual_units,
            filters=filters,
            strides=strides,
            activation=lrelu,
            mode=mode)

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(
                net_output_ops)})

    # 2. set up a loss function
    if loss_type == 'ce':
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=net_output_ops['logits'], labels=labels['y'])
        loss = tf.reduce_mean(ce)
    elif loss_type == 'balce':
        loss = sparse_balanced_crossentropy(
            net_output_ops['logits'], labels['y'])

    # 3. define a training op and ops for updating
    # moving averages (i.e. for batch normalisation)
    global_step = tf.train.get_global_step()
    if params["opt"] == 'adam':
        optimiser = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"], epsilon=1e-5)
    elif params["opt"] == 'momentum':
        optimiser = tf.train.MomentumOptimizer(
            learning_rate=params["learning_rate"], momentum=0.9)
    elif params["opt"] == 'rmsprop':
        optimiser = tf.train.RMSPropOptimizer(
            learning_rate=params["learning_rate"], momentum=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)

    # 4.1 (optional) create custom image summaries for tensorboard
    my_image_summaries = {}
    my_image_summaries['feat_t1'] = tf.expand_dims(
        features['x'][:, 0, :, :, 0], 3)
    my_image_summaries['labels'] = tf.expand_dims(
        tf.cast(labels['y'], tf.float32)[:, 0, :, :], 3)
    my_image_summaries['predictions'] = tf.expand_dims(
        tf.cast(net_output_ops['y_'], tf.float32)[:, 0, :, :], 3)

    [tf.summary.image(name, image)
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) create custom metric summaries for tensorboard
    dice_tensor = tf.py_func(
        dice, [net_output_ops['y_'], labels['y'],
               tf.constant(NUM_CLASSES)], tf.float32)

    [tf.summary.scalar('dsc_l{}'.format(i), dice_tensor[i])
     for i in range(NUM_CLASSES)]

    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=net_output_ops,
        loss=loss, train_op=train_op,
        eval_metric_ops=None)


def train(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')

    with open(args.config) as f:
        run_config = json.load(f)

    # Parse csv files for file names
    train_filenames = pd.read_csv(
        args.train_csv, dtype=object, keep_default_na=False,
        na_values=[]).as_matrix()

    val_filenames = pd.read_csv(
        args.val_csv, dtype=object, keep_default_na=False,
        na_values=[]).as_matrix()

    # Set up a data reader to handle the file i/o.
    reader_params = {
        'n_examples': 32,
        'example_size': [64, 64, 64],
        'extract_examples': True
    }

    reader_example_shapes = {
        'features': {'x': reader_params['example_size'] + [NUM_CHANNELS, ]},
        'labels': {'y': reader_params['example_size']}}

    reader = Reader(read_fn, {'features': {'x': tf.float32},
                              'labels': {'y': tf.int32}})

    # Get input functions and queue initialisation hooks
    # for training and validation data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        train_filenames,
        tf.estimator.ModeKeys.TRAIN,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        val_filenames,
        tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=min(SHUFFLE_CACHE_SIZE, EVAL_STEPS),
        params=reader_params)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.save_path,
        params=run_config,
        config=tf.estimator.RunConfig(session_config=config))

    # Hooks for validation summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(args.save_path, 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(
        every_n_steps=EVAL_EVERY_N_STEPS, output_dir=args.save_path)

    print('Starting training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn=train_input_fn,
                hooks=[train_qinit_hook, step_cnt_hook],
                steps=EVAL_EVERY_N_STEPS)

            results_val = nn.evaluate(
                input_fn=val_input_fn,
                hooks=[val_qinit_hook, val_summary_hook],
                steps=EVAL_STEPS)
            print('Step = {}; val loss = {:.5f};'.format(
                results_val['global_step'], results_val['loss']))

    except KeyboardInterrupt:
        pass

    print('Stopping now.')
    export_dir = nn.export_savedmodel(
        export_dir_base=args.save_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(reader_example_shapes))
    print('Model saved to {}.'.format(export_dir))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Example: Synapse CT example segmentation training script')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--save_path', '-p', default='/tmp/synapse_ct_seg/')
    parser.add_argument('--train_csv', default='train.csv')
    parser.add_argument('--val_csv', default='val.csv')
    parser.add_argument('--config', default="config.json")

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Create model save path
    os.system("rm -rf %s" % args.save_path)
    os.system("mkdir -p %s" % args.save_path)

    # Call training
    train(args)
