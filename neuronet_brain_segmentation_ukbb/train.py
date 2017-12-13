# -*- coding: utf-8 -*-
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from dltk.core.metrics import *
from dltk.core.losses import *
from dltk.core.activations import leaky_relu
from dltk.io.abstract_reader import Reader

from neuronet import neuronet_3d

from reader import read_fn
import json

# PARAMS
EVAL_EVERY_N_STEPS = 1000
EVAL_STEPS = 10

NUM_CHANNELS = 1

BATCH_SIZE = 1
SHUFFLE_CACHE_SIZE = 16

MAX_STEPS = 100000


# MODEL
def model_fn(features, labels, mode, params):

    # 1. create a model and its outputs    
    def lrelu(x):
        return leaky_relu(x, 0.1)
    
    protocols = params["protocols"]
   
    net_output_ops = neuronet_3d(features['x'],
                                 num_classes=params["num_classes"],
                                 protocols=protocols,
                                 num_res_units=params["network"]["num_residual_units"],
                                 filters=params["network"]["filters"],
                                 strides=params["network"]["strides"],
                                 activation=lrelu,
                                 mode=mode)
    
    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})
    
    # 2. set up a loss function
    ce = []
    for p in protocols:
        ce.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=net_output_ops['logits_{}'.format(p)],
                labels=labels[p])))

    # Sum the crossentropy losses and divide through number of protocols to be predicted 
    loss = tf.div(tf.add_n(ce), tf.constant(len(protocols), dtype=tf.float32))
    
    # 3. define a training op and ops for updating moving averages (i.e. for batch normalisation)  
    global_step = tf.train.get_global_step()
    optimiser = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], epsilon=1e-5)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)
    
    # 4.1 (optional) create custom image summaries for tensorboard
    my_image_summaries = {}
    my_image_summaries['feat_t1'] = features['x'][0,64,:,:,0]
    for p in protocols:
        my_image_summaries['{}/lbl'.format(p)] = tf.cast(labels[p], tf.float32)[0,64,:,:]
        my_image_summaries['{}/pred'.format(p)] = tf.cast(net_output_ops['y_{}'.format(p)], tf.float32)[0,64,:,:]
        
    expected_output_size = [1, 128, 128, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]
    
    # 4.2 (optional) create custom metric summaries for tensorboard 
    for i in range(len(protocols)):
        p = protocols[i]
        c = tf.constant(params["num_classes"][i])
        
        mean_dice = tf.reduce_mean(tf.py_func(dice, [net_output_ops['y_{}'.format(p)], labels[p], c], tf.float32)[1:])
        tf.summary.scalar('dsc_{}'.format(p), mean_dice)
        
    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode, predictions=None, loss=loss, train_op=train_op, eval_metric_ops=None)


def train(args, config):

    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')
    # Parse csv files for file names
    train_filenames = pd.read_csv(args.train_csv,
                                  dtype=object,
                                  keep_default_na=False,
                                  na_values=[]).as_matrix()
    
    val_filenames = pd.read_csv(args.val_csv, 
                                dtype=object,
                                keep_default_na=False,
                                na_values=[]).as_matrix()
    
    # Set up a data reader to handle the file i/o. 
    reader_params = {
        'n_examples': 8,
        'example_size': [128, 128, 128],
        'extract_examples': True,
        'protocols': config["protocols"]}
    
    reader_example_shapes = {
        'features': {'x': reader_params['example_size'] + [NUM_CHANNELS,]},
        'labels': {p: reader_params['example_size'] for p in config["protocols"]}}
    
    reader = Reader(read_fn,
                    {'features': {'x': tf.float32}, 
                    'labels': {p: tf.int32 for p in config["protocols"]}})

    # Get input functions and queue initialisation hooks for training and validation data
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
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)
    
    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(model_fn=model_fn,
                                model_dir=config["model_path"],
                                params=config, 
                                config=tf.estimator.RunConfig(session_config=tf.ConfigProto()))
    
    # Hooks for validation summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(config["model_path"], 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(
        every_n_steps=EVAL_EVERY_N_STEPS, output_dir=config["model_path"])
    
    print('Starting training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(input_fn=train_input_fn,
                     hooks=[train_qinit_hook, step_cnt_hook], 
                     steps=EVAL_EVERY_N_STEPS)
            
            results_val = nn.evaluate(input_fn=val_input_fn, 
                                      hooks=[val_qinit_hook, val_summary_hook],
                                      steps=EVAL_STEPS)
            print('Step = {}; val loss = {:.5f};'.format(results_val['global_step'], results_val['loss']))

    except KeyboardInterrupt:
        pass
    
    print('Stopping now.')
    export_dir = nn.export_savedmodel(
        export_dir_base=config["model_path"],
        serving_input_receiver_fn=reader.serving_input_receiver_fn(reader_example_shapes))
    print('Model saved to {}.'.format(export_dir))

        
if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='UK Biobank NeuroNet training')
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    
    #parser.add_argument('--model_path', '-p', default='/tmp/synapse_ct_seg/')
    parser.add_argument('--train_csv', default='train.csv')
    parser.add_argument('--val_csv', default='val.csv')
    parser.add_argument('--config', default='config.json')
    
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
    
    # Parse the run config
    with open(args.config) as f:
        config = json.load(f)

    # Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(config["model_path"]))

    if not os.path.isdir(config["model_path"]):
        os.system('mkdir -p {}'.format(config["model_path"]))
    else:
        print('Resuming training on model_path {}'.format(config["model_path"]))

    # Call training
    train(args, config)