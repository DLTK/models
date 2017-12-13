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
import json

from tensorflow.contrib import predictor

from dltk.core import metrics as metrics
from dltk.utils import sliding_window_segmentation_inference

from reader import read_fn


def predict(args, config):

    # Read in the csv with the file names you would want to predict on
    file_names = pd.read_csv(args.csv,
                             dtype=object,
                             keep_default_na=False,
                             na_values=[]).as_matrix()

    # From the model model_path, parse the latest saved estimator model
    # and restore a predictor from it
    export_dir = [os.path.join(config["model_path"], o) for o in os.listdir(config["model_path"])
                  if os.path.isdir(os.path.join(config["model_path"], o)) and o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)
    
    protocols = config["protocols"]
    # Fetch the output probability ops of the trained network
    y_probs = [my_predictor._fetch_tensors['y_prob_{}'.format(p)] for p in protocols]
    num_classes = [yp.get_shape().as_list()[-1] for yp in y_probs]
    
    # Iterate through the files, predict on the full volumes and
    #  compute a Dice similariy coefficient
    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params={'extract_examples': False, 
                                  'protocols': config["protocols"]}):

        print('Running file {}'.format(output['img_id']))
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension
        #  as required
        img = np.expand_dims(output['features']['x'], axis=0)
        lbls = [np.expand_dims(output['labels'][p], axis=0) for p in protocols]
        
        print('Image shape {}'.format(img.shape))

        # Do a sliding window inference with our DLTK wrapper
        preds = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=y_probs,
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=2)[0]

        # Calculate the prediction from the probabilities
        preds = [np.argmax(pred, -1) for pred in preds]

        # Save the file as .nii.gz using the header information from the
        # original sitk image
        out_folder = os.path.join(config["out_segm_path"], '{}'.format(output['img_id']))
        os.system('mkdir -p {}'.format(out_folder))
        
        for i in range(len(protocols)): 
            output_fn = os.path.join(out_folder, protocols[i] + '.nii.gz')
            new_sitk = sitk.GetImageFromArray(preds[i].astype(np.int32))
            new_sitk.CopyInformation(output['sitk'])
            sitk.WriteImage(new_sitk, output_fn)

        # Print outputs
        print('ID={}; input_dim={}; time={};'.format(
            output['img_id'], img.shape, time.time()-t0))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='iFind2 fetal segmentation deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    #parser.add_argument('--model_path', '-p', default='/tmp/fetal_segmentation/')
    parser.add_argument('--csv', default='test.csv')
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
        
    # Call training
    predict(args, config)
