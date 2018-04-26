# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk

from tensorflow.contrib import predictor

from dltk.core import metrics as metrics

from dltk.utils import sliding_window_segmentation_inference


from reader import read_fn

READER_PARAMS = {'extract_examples': False}


def predict(args):
    # Read in the csv with the file names you would want to predict on
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    print('Loading from {}'.format(args.model_path))
    my_predictor = predictor.from_saved_model(args.model_path)

    # Fetch the output probability op of the trained network
    y_prob = my_predictor._fetch_tensors['y_prob']
    print('Got y_prob as {}'.format(y_prob))
    num_classes = y_prob.get_shape().as_list()[-1]

    mode = (tf.estimator.ModeKeys.PREDICT if args.predict_only
            else tf.estimator.ModeKeys.EVAL)

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    for output in read_fn(file_references=file_names,
                          mode=mode,
                          params=READER_PARAMS):
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension as
        # required
        img = np.expand_dims(output['features']['x'], axis=0)

        print('running inference on {} with img {} and op {}'.format(
            my_predictor._feed_tensors['x'], img.shape, y_prob))
        # Do a sliding window inference with our DLTK wrapper
        pred = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[y_prob],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=32)[0]

        # Calculate the prediction from the probabilities
        pred = np.argmax(pred, -1)

        if not args.predict_only:
            lbl = np.expand_dims(output['labels']['y'], axis=0)
            # Calculate the Dice coefficient
            dsc = metrics.dice(pred, lbl, num_classes)[1:].mean()

        # Save the file as .nii.gz using the header information from the
        # original sitk image
        output_fn = os.path.join(
            args.export_path, '{}_seg.nii.gz'.format(output['img_name']))

        new_sitk = sitk.GetImageFromArray(pred[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])

        sitk.WriteImage(new_sitk, output_fn)

        if args.predict_only:
            print('Id={}; time={:0.2} secs; output_path={};'.format(
                output['img_name'], time.time() - t0, output_fn))
        else:
            # Print outputs
            print(
                'Id={}; Dice={:0.4f} time={:0.2} secs; output_path={};'.format(
                    output['img_name'], dsc, time.time() - t0, output_fn))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Synapse MultiAtlas example segmentation deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--predict_only', '-n', default=False,
                        action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p', default='/tmp/synapse_ct_seg/')
    parser.add_argument('--export_path', '-e', default='/tmp/synapse_ct_seg/')
    parser.add_argument('--csv', default='train.csv')

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

    # Call training
    predict(args)
